//! The interpolate routine trait, the tensors a launch binds, and the launch of one routine.

use crate::definition::{
    InterpolateForwardProblem, InterpolateMode, InterpolateOptions, mode_properties,
};
use crate::kernels::InterpolateRoutineProblem;
use crate::kernels::filter::{
    BicubicFilter, BilinearFilter, Lanczos3Filter, NearestFilter, SeparableFilterFamily,
};
use crate::kernels::space::{self, BATCH, CHANNEL, OUTPUT_H, OUTPUT_W};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Bound, BoundaryPolicy, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Partitioning};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid, Routine};

/// A routine of this kind: it prepares a launch of any filter and launches it on tensors, its
/// kernel compiled for filter `F`.
pub trait InterpolateRoutine: Routine<Problem = InterpolateRoutineProblem> {
    /// Launch the kernel `blueprint` compiles for `F`, on `grid`, over `tensors`.
    fn dispatch<F: SeparableFilterFamily>(
        &self,
        tensors: &InterpolateTensors,
        blueprint: Self::Blueprint,
        grid: Grid,
    );
}

/// `tensors` resampled by `routine` at `settings` on `hw`.
///
/// # Errors
///
/// Why the routine cannot run the problem here.
pub fn launch_interpolate_routine<R: InterpolateRoutine>(
    routine: &R,
    hw: &Hardware,
    tensors: &InterpolateTensors,
    settings: &R::Settings,
) -> Result<(), Refusal> {
    let (blueprint, grid) = routine.prepare(hw, tensors.problem(), settings)?;
    match tensors.problem().forward.options.mode {
        InterpolateMode::Nearest(_) => routine.dispatch::<NearestFilter>(tensors, blueprint, grid),
        InterpolateMode::Bilinear => routine.dispatch::<BilinearFilter>(tensors, blueprint, grid),
        InterpolateMode::Bicubic => routine.dispatch::<BicubicFilter>(tensors, blueprint, grid),
        InterpolateMode::Lanczos3 => routine.dispatch::<Lanczos3Filter>(tensors, blueprint, grid),
    }
    Ok(())
}

/// The NHWC input and output of one interpolation, and the problem they state.
#[derive(Clone)]
pub struct InterpolateTensors {
    client: Client,
    input: TensorBinding,
    output: TensorBinding,
    problem: InterpolateRoutineProblem,
}

impl InterpolateTensors {
    /// `input` resampled into `output`, both NHWC in `element`, as `options` say.
    pub fn new(
        client: &Client,
        input: TensorBinding,
        output: TensorBinding,
        options: InterpolateOptions,
        element: ElemType,
    ) -> Self {
        let problem = InterpolateRoutineProblem {
            forward: InterpolateForwardProblem::from_input_output_shapes(
                &input.shape,
                &[output.shape[1], output.shape[2]],
                options,
            ),
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

    pub fn problem(&self) -> &InterpolateRoutineProblem {
        &self.problem
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Both operands bound on `grid` at `width`, placed by `partitioning`: the input gathered over
    /// every output position's taps, its reads masked where `checked`.
    pub fn bind(
        &self,
        partitioning: &Partitioning,
        width: usize,
        checked: bool,
        grid: &Grid,
    ) -> InterpolateBinding {
        let launcher = grid.launcher(&self.client, || {
            (partitioning.clone(), partitioning.space().clone())
        });
        let problem = &self.problem;
        let boundary = match checked {
            true => BoundaryPolicy::Every(mode_properties(problem.forward.options.mode).boundary),
            false => BoundaryPolicy::Unchecked,
        };
        let input = launcher
            .arg(self.input.clone(), problem.element)
            .gathered(space::input_projection(
                problem.row(),
                problem.col(),
                problem.radius(),
            ))
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
        InterpolateBinding {
            launcher,
            input,
            output,
        }
    }
}

/// The operands of one launch, bound, and the launcher they were bound through.
pub struct InterpolateBinding {
    pub launcher: Launcher,
    pub input: Bound,
    pub output: Bound,
}
