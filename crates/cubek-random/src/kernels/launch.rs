//! The random routine trait, the tensor a launch binds, and the launch of one routine.

use crate::kernels::{RandomRoutineProblem, VALUES};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::{LaunchArg, TensorBinding};
use cubecl::tile::launch::{Bound, Refusal};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Partitioning};
use cubecl::zspace::{Shape, Strides, Tiling};
use cubek_microkernel::random::{Distribution, SeedsLaunch};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid, Routine};

/// A routine of this kind: it prepares a launch of any distribution and launches it on an output,
/// its kernel compiled for distribution `D`.
pub trait RandomRoutine: Routine<Problem = RandomRoutineProblem> {
    /// Launch the kernel `blueprint` compiles for `D`, on `grid`, filling `tensor` with draws of
    /// `args` off `seeds`.
    fn dispatch<D: Distribution>(
        &self,
        tensor: &RandomTensor,
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
) -> Result<(), Refusal> {
    let (blueprint, grid) = routine.prepare(hw, tensor.problem(), settings)?;
    routine.dispatch::<D>(tensor, distribution.args(), seeds, blueprint, grid);
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
    pub fn bind(&self, partitioning: &Partitioning, width: usize, grid: &Grid) -> RandomBinding {
        let launcher = grid.launcher(&self.client, || {
            (
                partitioning.clone().all_dynamic(),
                partitioning.space().clone(),
            )
        });
        let output = launcher
            .arg(self.output.clone(), self.problem.element)
            .axes(&vec![VALUES; self.output.shape.len()])
            .vectorize(width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        RandomBinding { launcher, output }
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
