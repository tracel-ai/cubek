//! The random routine trait, the tensor a launch binds, and the launch of one routine.

use crate::kernels::{RandomRoutineProblem, VALUES};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::{LaunchArg, TensorBinding};
use cubecl::tile::launch::{Bound, Decline};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Partitioning};
use cubecl::zspace::{Shape, Strides, Tiling};
use cubek_microkernel::random::{Distribution, SeedsLaunch};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable, Grid, Routine};

/// A routine of this kind: it prepares a launch of any distribution and launches it on an output,
/// its kernel compiled for distribution `D`.
///
/// # Assumptions
///
/// [`dispatch`](Self::dispatch) is handed the binding [`bind`](Self::bind) returned for the same
/// tensor, blueprint and grid.
///
/// # Guarantees
///
/// [`dispatch`](Self::dispatch) launches whatever it is handed: every refusal of a launch is met
/// while binding, before its kernel launches.
pub trait RandomRoutine: Routine<Problem = RandomRoutineProblem> {
    /// `tensor` bound as `blueprint` states it, on `grid`.
    ///
    /// # Errors
    ///
    /// As [`RandomTensor::bind`].
    fn bind(
        &self,
        tensor: &RandomTensor,
        blueprint: &Self::Blueprint,
        grid: &Grid,
    ) -> Result<RandomBinding, Decline>;

    /// Launch the kernel `blueprint` compiles for `D`, on `grid`, filling `binding`, `tensor` as
    /// [`bind`](Self::bind) bound it, with draws of `args` off `seeds`.
    fn dispatch<D: Distribution>(
        &self,
        tensor: &RandomTensor,
        binding: RandomBinding,
        args: <D::Args as LaunchArg>::RuntimeArg,
        seeds: SeedsLaunch,
        blueprint: Self::Blueprint,
        grid: Grid,
    );
}

/// `tensor` filled by `routine` at `settings` on `hw` with draws of `distribution` off `seeds`.
///
/// # Errors
///
/// Why the routine cannot run the problem here.
pub fn launch_random_routine<R: RandomRoutine, D: Distribution>(
    routine: &R,
    hw: &Hardware,
    tensor: &RandomTensor,
    settings: &R::Settings,
    distribution: D,
    seeds: SeedsLaunch,
) -> Result<(), Decline> {
    let (blueprint, grid) = R::prepare(hw, tensor.problem(), settings)?;
    let binding = routine.bind(tensor, &blueprint, &grid)?;
    routine.dispatch::<D>(tensor, binding, distribution.args(), seeds, blueprint, grid);
    Ok(())
}

/// The output of one draw, and the problem it states.
#[derive(Clone)]
pub struct RandomTensor {
    client: Client,
    output: TensorBinding,
    problem: RandomRoutineProblem,
}

impl RandomTensor {
    /// `output`, stored in `element`, filled with draws `lines_per_draw` lines at a time.
    pub fn new(
        client: &Client,
        output: TensorBinding,
        element: ElemType,
        lines_per_draw: usize,
    ) -> Self {
        let output = flattened(output);
        let problem = RandomRoutineProblem {
            values: output.shape.iter().product(),
            element,
            lines_per_draw,
            alignment: ElemAlignment::new(&output.shape, &output.strides),
        };
        Self {
            client: client.clone(),
            output,
            problem,
        }
    }

    pub fn problem(&self) -> &RandomRoutineProblem {
        &self.problem
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The output bound on `grid` at `width`, every dim of it projected onto the values, placed
    /// by `partitioning`, which the kernel compiles with the values left to the launch.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared.
    pub fn bind(
        &self,
        partitioning: &Partitioning,
        width: usize,
        grid: &Grid,
    ) -> Result<RandomBinding, Decline> {
        let launcher = grid.launcher(&self.client, || {
            (
                partitioning.clone().all_dynamic(),
                partitioning.space().clone(),
            )
        })?;
        let output = launcher
            .arg(self.output.clone(), self.problem.element)
            .dimensions(&vec![VALUES; self.output.shape.len()])
            .vectorize(width)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        Ok(RandomBinding { launcher, output })
    }
}

/// `output` as one dim when its values fill a run of the buffer with no gap, in any order of its
/// dims: a draw does not care where a value lies, and one dim is addressed as it is counted, with
/// no division per write to recover the coordinates of every dim. Any other output keeps its dims.
fn flattened(output: TensorBinding) -> TensorBinding {
    let mut dims: Vec<(usize, usize)> = output
        .shape
        .iter()
        .zip(output.strides.iter())
        .map(|(&size, &stride)| (stride, size))
        .filter(|&(_, size)| size != 1)
        .collect();
    dims.sort_unstable();
    let mut run = 1;
    for &(stride, size) in &dims {
        if stride != run {
            return output;
        }
        run *= size;
    }
    if output.tiling != Tiling::UNTILED || output.shape.len() == 1 {
        return output;
    }
    let values = output.shape.iter().product();
    TensorBinding {
        shape: Shape::new([values]),
        strides: Strides::new(&[1]),
        ..output
    }
}

/// The output of one launch, bound, and the launcher it was bound through.
pub struct RandomBinding {
    pub launcher: Launcher,
    pub output: Bound,
}
