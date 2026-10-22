use crate::Hardware;
use crate::microkernel::{CubeFootprint, CubePlanes, KernelEstimate};
use crate::routine::{ADMITTED_PLAN_RUNS, LaunchSettings};
use cubecl::CubeCount;
use cubecl::client::Client;
use cubecl::tile::launch::Grid as LaunchGrid;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Launcher, Partitioning, Space};

/// The cubes a launch spawns, the units each holds, and the shared memory each holds.
///
/// Beside a blueprint rather than inside it: it is read off the partitioning at the plane width
/// this device runs, so a blueprint replayed on another device spawns that device's grid.
#[derive(Debug, Clone)]
pub struct Grid {
    /// The cubes and their units, checked against the device as every family's launch is.
    pub launch: LaunchSettings,
    /// Bytes of shared memory one cube holds.
    pub shared_memory: usize,
}

/// What one cube of a launch asks of the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeResources {
    /// Units in one plane: the cube's `x`.
    pub plane_width: u32,
    /// Bytes of shared memory one cube holds.
    pub shared_memory: usize,
}

impl Grid {
    /// The launcher a kernel compiled for `kernel` runs on over `concrete`'s real extents, spawning
    /// this grid: the cubes a launch's own partitioning states, not ones cubek derives again.
    ///
    /// # Panics
    ///
    /// Where the device refuses the grid, which [`Grid::new`] already admitted for it.
    pub fn launcher(&self, client: &Client, kernel: Partitioning, concrete: &Space) -> Launcher {
        Launcher::new(
            client,
            kernel,
            concrete,
            LaunchGrid::Stated {
                cube_count: self.launch.cube_count.clone(),
                cube_dim: self.launch.cube_dim,
            },
        )
        .and_then(|launcher| launcher.holding(self.shared_memory))
        .expect(ADMITTED_PLAN_RUNS)
    }

    /// The grid `partitioning` distributes to on this device, each cube holding `cube`.
    ///
    /// # Errors
    ///
    /// A cube or a grid this device does not spawn, one that spawns nothing, or a cube holding
    /// more shared memory than the device gives one.
    pub fn new(
        hw: &Hardware,
        partitioning: &Partitioning,
        cube: CubeResources,
    ) -> Result<Self, Refusal> {
        let launch = LaunchSettings {
            cube_dim: partitioning.cube_dim(cube.plane_width),
            cube_count: partitioning.cube_count(),
            vectors: (),
        }
        .validate(hw)?;
        let available = hw.shared_memory_budget();
        if cube.shared_memory > available {
            return Err(Refusal::SharedMemoryPastDevice {
                bytes: cube.shared_memory,
                most: available,
            });
        }
        Ok(Self {
            launch,
            shared_memory: cube.shared_memory,
        })
    }

    /// Cubes the grid spawns; `None` where the count is only known at launch.
    pub fn cubes(&self) -> Option<u64> {
        match self.launch.cube_count {
            CubeCount::Static(x, y, z) => Some(u64::from(x) * u64::from(y) * u64::from(z)),
            CubeCount::Dynamic(_) => None,
        }
    }

    /// The planes of one cube.
    pub fn planes(&self) -> CubePlanes {
        CubePlanes {
            count: self.launch.cube_dim.y as usize,
            width: self.launch.cube_dim.x as usize,
        }
    }
}

impl From<&Grid> for KernelEstimate {
    /// The estimate of a kernel on `grid`, before any owner has set its part.
    fn from(grid: &Grid) -> Self {
        Self {
            cubes: grid.cubes(),
            cube: CubeFootprint {
                units: grid.launch.units_per_cube(),
                shared_memory: grid.shared_memory,
            },
            demand: None,
            staging: None,
            work: None,
        }
    }
}
