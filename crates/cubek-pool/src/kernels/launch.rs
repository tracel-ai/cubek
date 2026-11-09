//! The pool routine trait, the tensors a launch binds, and the launch of one routine.

use crate::definition::{MaxPoolOptions, PoolWindow};
use crate::kernels::PoolRoutineProblem;
use crate::kernels::space::{self, BATCH, CHANNEL, OUTPUT_H, OUTPUT_W, WindowAxis};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{Bound, BoundaryPolicy, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Partitioning};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid, Routine};

/// A routine of this kind: it prepares a launch of a max pooling and launches it on tensors.
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

/// The NHWC input and output of one max pooling, and the problem they state.
#[derive(Clone)]
pub struct PoolTensors {
    client: Client,
    input: TensorBinding,
    output: TensorBinding,
    problem: PoolRoutineProblem,
}

impl PoolTensors {
    /// `input` max pooled into `output`, both NHWC in `element`, as `options` say.
    pub fn new(
        client: &Client,
        input: TensorBinding,
        output: TensorBinding,
        options: MaxPoolOptions<2>,
        element: ElemType,
    ) -> Self {
        let problem = PoolRoutineProblem {
            batch: input.shape[0],
            input: [input.shape[1], input.shape[2]],
            output: [output.shape[1], output.shape[2]],
            channels: input.shape[3],
            windows: window_axes(&options.window, options.dilation),
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
    /// every output position's window, its reads masked where `checked`, past which the reduce
    /// reads the max's identity.
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
