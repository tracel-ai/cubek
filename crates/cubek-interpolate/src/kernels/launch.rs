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
use cubecl::tile::launch::{Bound, BoundaryPolicy, Decline};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Partitioning};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable, Grid, Routine};

/// A routine of this kind: it prepares a launch of any filter and launches it on tensors, its
/// kernel compiled for filter `F`.
///
/// # Assumptions
///
/// [`dispatch`](Self::dispatch) is handed the binding [`bind`](Self::bind) returned for the same
/// tensors, blueprint and grid.
///
/// # Guarantees
///
/// [`dispatch`](Self::dispatch) launches whatever it is handed: every refusal of a launch is met
/// while binding, before its kernel launches.
pub trait InterpolateRoutine: Routine<Problem = InterpolateRoutineProblem> {
    /// `tensors`' operands bound as `blueprint` states them, on `grid`.
    ///
    /// # Errors
    ///
    /// As [`InterpolateTensors::bind`].
    fn bind(
        &self,
        tensors: &InterpolateTensors,
        blueprint: &Self::Blueprint,
        grid: &Grid,
    ) -> Result<InterpolateBinding, Decline>;

    /// Launch the kernel `blueprint` compiles for `F`, on `grid`, over `binding`, `tensors`'
    /// operands as [`bind`](Self::bind) bound them.
    fn dispatch<F: SeparableFilterFamily>(
        &self,
        tensors: &InterpolateTensors,
        binding: InterpolateBinding,
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
) -> Result<(), Decline> {
    let (blueprint, grid) = R::prepare(hw, tensors.problem(), settings)?;
    let binding = routine.bind(tensors, &blueprint, &grid)?;
    match tensors.problem().forward.options.mode {
        InterpolateMode::Nearest(_) => {
            routine.dispatch::<NearestFilter>(tensors, binding, blueprint, grid)
        }
        InterpolateMode::Bilinear => {
            routine.dispatch::<BilinearFilter>(tensors, binding, blueprint, grid)
        }
        InterpolateMode::Bicubic => {
            routine.dispatch::<BicubicFilter>(tensors, binding, blueprint, grid)
        }
        InterpolateMode::Lanczos3 => {
            routine.dispatch::<Lanczos3Filter>(tensors, binding, blueprint, grid)
        }
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
    ///
    /// # Assumptions
    ///
    /// `input` and `output` are both of rank four, as [`interpolate`](crate::interpolate) checks.
    ///
    /// # Panics
    ///
    /// On an `input` or an `output` of another rank, against its Assumption.
    pub fn new(
        client: &Client,
        input: TensorBinding,
        output: TensorBinding,
        options: InterpolateOptions,
        element: ElemType,
    ) -> Self {
        assert!(
            input.shape.len() == 4 && output.shape.len() == 4,
            "the input and the output are NHWC, of rank four"
        );
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
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared.
    pub fn bind(
        &self,
        partitioning: &Partitioning,
        width: usize,
        checked: bool,
        grid: &Grid,
    ) -> Result<InterpolateBinding, Decline> {
        let launcher = grid.launcher(&self.client, || {
            (partitioning.clone(), partitioning.space().clone())
        })?;
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
            .or_broken(ADMITTED_PLAN_RUNS)?;
        let output = launcher
            .arg(self.output.clone(), problem.element)
            .dimensions(&[BATCH, OUTPUT_H, OUTPUT_W, CHANNEL])
            .vectorize(width)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        Ok(InterpolateBinding {
            launcher,
            input,
            output,
        })
    }
}

/// The operands of one launch, bound, and the launcher they were bound through.
pub struct InterpolateBinding {
    pub launcher: Launcher,
    pub input: Bound,
    pub output: Bound,
}
