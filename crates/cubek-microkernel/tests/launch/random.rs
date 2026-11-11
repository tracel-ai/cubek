//! The random microkernels on one work item: one unit's run of `values`, or one plane's, its lanes
//! interleaved, each value drawn into a contiguous output.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder, UnitCube};
use cubecl::tile::{
    Axis, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::random::plane::{
    PlaneRandomMicrokernel, PlaneRandomMicrokernelSettings, plane_random_microkernel,
};
use cubek_microkernel::random::unit::{
    UnitRandomMicrokernel, UnitRandomMicrokernelSettings, unit_random_microkernel,
};
use cubek_microkernel::random::{Distribution, DistributionArgs, Seeds, SeedsLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The output's one axis.
const VALUES: Axis = Axis(0);

/// Which random microkernel draws the work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomMicrokernel {
    /// One unit's run, drawn line after line.
    Unit(UnitRandomMicrokernel),
    /// One plane's run, every lane drawing the lines `lane + i · lanes`.
    Plane(PlaneRandomMicrokernel),
}

/// One unit's run: `lines` lines, each `width` values wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitRun {
    pub lines: usize,
    pub width: usize,
}

/// A random microkernel ready to launch on one work item of `values` values, `width` a line.
#[derive(Debug, Clone, Copy)]
pub struct RandomLaunch {
    pub microkernel: RandomMicrokernel,
    pub values: usize,
    pub width: usize,
    /// Units a plane holds: the lanes of a plane's run.
    pub plane_width: usize,
}

impl RandomLaunch {
    /// One unit's `run`.
    ///
    /// # Errors
    ///
    /// Why the device runs no unit random microkernel.
    pub fn unit(hw: &Hardware, run: UnitRun) -> Result<Self, Refusal> {
        let UnitRun { lines, width } = run;
        let microkernel = UnitRandomMicrokernel::best(hw, &(), &UnitRandomMicrokernelSettings)?;
        Ok(Self {
            microkernel: RandomMicrokernel::Unit(microkernel),
            values: lines * width,
            width,
            plane_width: hw.dispatch_plane_size() as usize,
        })
    }

    /// One plane's run at `settings`, `width` values a line.
    ///
    /// # Errors
    ///
    /// Why the device runs no plane random microkernel.
    pub fn plane(
        hw: &Hardware,
        settings: &PlaneRandomMicrokernelSettings,
        width: usize,
    ) -> Result<Self, Refusal> {
        let microkernel = PlaneRandomMicrokernel::best(hw, &(), settings)?;
        let plane_width = hw.dispatch_plane_size() as usize;
        Ok(Self {
            microkernel: RandomMicrokernel::Plane(microkernel),
            values: plane_width * microkernel.values_per_lane,
            width,
            plane_width,
        })
    }

    /// Draws of `D` at `args` off `seeds` into `out`, `values` values of `element` in a row.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch<D: Distribution>(
        &self,
        client: &Client,
        out: TensorArg,
        element: ElemType,
        args: <<D as DistributionArgs>::Args as LaunchArg>::RuntimeArg,
        seeds: SeedsLaunch,
    ) -> Result<(), Refusal> {
        let space = Space::new(&[(VALUES, self.values)]);
        let run = Levels::leaf(&[(VALUES, self.values)]);
        let levels = match self.microkernel {
            RandomMicrokernel::Unit(_) => run.launch_units(
                &[VALUES],
                UnitCube {
                    plane_width: self.plane_width,
                    planes: 1,
                },
            ),
            RandomMicrokernel::Plane(_) => run.launch_planes(
                &[VALUES],
                PlaneOrder::new(&[(VALUES, 1)], CubeOrder::RowMajor),
            ),
        };
        let partitioning = Partitioning::new(space.clone(), levels.build());
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let out = TileTensorLaunch::new(out, element, self.width, TileSpec::direct(&[VALUES]));
        // SAFETY: the output is bound through the launcher at the work item's own extent, and the
        // kernel indexes no buffer but through its tile; it holds no shared memory.
        unsafe {
            match self.microkernel {
                RandomMicrokernel::Unit(microkernel) => unit_random_kernel::launch_unchecked::<D>(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    out,
                    launcher.partitioning_arg(),
                    args,
                    seeds,
                    microkernel,
                    element,
                ),
                RandomMicrokernel::Plane(microkernel) => {
                    plane_random_kernel::launch_unchecked::<D>(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        out,
                        launcher.partitioning_arg(),
                        args,
                        seeds,
                        microkernel,
                        element,
                    )
                }
            }
        }
        Ok(())
    }
}

type Args<D> = <D as DistributionArgs>::Args;

/// The one unit drawing its run.
#[cube(launch_unchecked)]
fn unit_random_kernel<D: Distribution, E: Numeric, N: Size>(
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    args: Args<D>,
    seeds: Seeds,
    #[comptime] microkernel: UnitRandomMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let o = out.tile(&partitioning);
    for unit in &partitioning {
        let mut run = o.at(&unit);
        unit_random_microkernel::<D, E, N>(&mut run, &args, &seeds, microkernel);
    }
}

/// The one plane drawing its run.
#[cube(launch_unchecked)]
fn plane_random_kernel<D: Distribution, E: Numeric, N: Size>(
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    args: Args<D>,
    seeds: Seeds,
    #[comptime] microkernel: PlaneRandomMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let o = out.tile(&partitioning);
    for plane in &partitioning {
        let mut run = o.at(&plane);
        plane_random_microkernel::<D, E, N>(&mut run, &args, &seeds, microkernel);
    }
}
