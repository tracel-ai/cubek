//! The pool routine trait, the tensors a launch binds, and the launch of one routine.

use crate::definition::{AvgPoolOptions, MaxPoolOptions, PoolWindow};
use crate::kernels::space::{self, BATCH, CHANNEL, OUTPUT_H, OUTPUT_W, WindowAxis};
use crate::kernels::{AverageCount, PoolRoutineProblem};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{Bound, BoundaryPolicy, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Partitioning};
use cubek_microkernel::pool::PoolReduction;
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid, Routine};

/// A routine of this kind: it prepares a launch of a pooling and launches it on tensors.
pub trait PoolRoutine: Routine<Problem = PoolRoutineProblem> {
    /// Launch the kernel `blueprint` compiles, on `grid`, over `tensors`.
    fn dispatch(&self, tensors: &PoolTensors, blueprint: Self::Blueprint, grid: Grid);
}

/// `tensors` pooled by `routine` at `settings` on `hw`.
///
/// # Errors
///
/// Why the routine cannot run the problem here.
pub fn launch_pool_routine<R: PoolRoutine>(
    routine: &R,
    hw: &Hardware,
    tensors: &PoolTensors,
    settings: &R::Settings,
) -> Result<(), Refusal> {
    let (blueprint, grid) = routine.prepare(hw, tensors.problem(), settings)?;
    routine.dispatch(tensors, blueprint, grid);
    Ok(())
}

/// What a pooling makes of its windows, as the caller states it.
#[derive(Debug, Clone)]
pub enum PoolOperation {
    Max(MaxPoolOptions<2>),
    Average(AvgPoolOptions<2>),
}

/// The NHWC input and output of one pooling, and the problem they state.
#[derive(Clone)]
pub struct PoolTensors {
    client: Client,
    input: TensorBinding,
    output: TensorBinding,
    problem: PoolRoutineProblem,
}

impl PoolTensors {
    /// `input` pooled into `output`, both NHWC in `element`, as `operation` says.
    pub fn new(
        client: &Client,
        input: TensorBinding,
        output: TensorBinding,
        operation: PoolOperation,
        element: ElemType,
    ) -> Self {
        let (window, dilation, reduction, count) = match operation {
            PoolOperation::Max(options) => {
                (options.window, options.dilation, PoolReduction::Max, None)
            }
            PoolOperation::Average(options) => {
                let count = match options.count_include_pad {
                    true => AverageCount::Padded,
                    false => AverageCount::InBounds,
                };
                (
                    options.window,
                    options.dilation,
                    PoolReduction::Average,
                    Some(count),
                )
            }
        };
        let problem = PoolRoutineProblem {
            batch: input.shape[0],
            input: [input.shape[1], input.shape[2]],
            output: [output.shape[1], output.shape[2]],
            channels: input.shape[3],
            windows: window_axes(&window, dilation),
            reduction,
            count,
            element,
            elem_alignments: [&input, &output]
                .map(|binding| ElemAlignment::new(&binding.shape, &binding.strides)),
        };
        Self {
            client: client.clone(),
            input,
            output,
            problem,
        }
    }

    pub fn problem(&self) -> &PoolRoutineProblem {
        &self.problem
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Both operands bound on `grid` at `width`, placed by `partitioning`: the input gathered over
    /// every output position's window, its reads masked where `checked`. A masked max reads the
    /// max's identity past the edge, an average zero.
    pub fn bind(
        &self,
        partitioning: &Partitioning,
        width: usize,
        checked: bool,
        grid: &Grid,
    ) -> PoolBinding {
        let launcher = grid.launcher(&self.client, || {
            (partitioning.clone(), partitioning.space().clone())
        });
        let problem = &self.problem;
        let boundary = match checked {
            true => BoundaryPolicy::Every(Boundary::Zero),
            false => BoundaryPolicy::Unchecked,
        };
        let [rows, cols] = problem.windows;
        let input = launcher
            .arg(self.input.clone(), problem.element)
            .gathered(space::input_projection(rows, cols))
            .boundary(boundary)
            .vectorize(width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let output = launcher
            .arg(self.output.clone(), problem.element)
            .axes(&[BATCH, OUTPUT_H, OUTPUT_W, CHANNEL])
            .vectorize(width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        PoolBinding {
            launcher,
            input,
            output,
        }
    }
}

/// The window along the rows, then the columns.
fn window_axes(window: &PoolWindow<2>, dilation: [usize; 2]) -> [WindowAxis; 2] {
    [0, 1].map(|axis| WindowAxis {
        taps: window.kernel_size[axis],
        stride: window.stride[axis],
        dilation: dilation[axis],
        padding: window.padding[axis],
    })
}

/// The operands of one launch, bound, and the launcher they were bound through.
pub struct PoolBinding {
    pub launcher: Launcher,
    pub input: Bound,
    pub output: Bound,
}
