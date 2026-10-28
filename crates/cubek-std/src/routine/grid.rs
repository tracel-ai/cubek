use crate::Hardware;
use crate::microkernel::{CubeFootprint, CubePlanes, KernelEstimate};
use crate::routine::{ADMITTED_PLAN_RUNS, LaunchSettings};
use cubecl::CubeCount;
use cubecl::client::Client;
use cubecl::tile::launch::Grid as LaunchGrid;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Launcher, Partitioning, Space};
use cubecl_environment::sync::{Arc, SyncOnceCell};

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
    /// The launcher the first launch on this grid built, shared by its clones.
    launcher: SettledLauncher,
}

/// A grid's launcher, built by its first launch and cloned by every later one: it reads nothing
/// but the grid and the blueprint the grid was prepared beside, and building it settles the
/// partitioning over the problem's extents, which costs more host time than a small product.
#[derive(Clone, Default)]
struct SettledLauncher(Arc<SyncOnceCell<Launcher>>);

impl core::fmt::Debug for SettledLauncher {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SettledLauncher")
    }
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
    /// The launcher a kernel compiled for the partitioning `settle` states runs on over the real
    /// extents of the space it states beside it, spawning this grid: the cubes a launch's own
    /// partitioning states, not ones cubek derives again. `settle` runs on the grid's first launch
    /// alone: every later one clones the launcher it built.
    ///
    /// # Panics
    ///
    /// Where the device refuses the grid, which [`Grid::new`] already admitted for it.
    pub fn launcher(
        &self,
        client: &Client,
        settle: impl FnOnce() -> (Partitioning, Space),
    ) -> Launcher {
        self.launcher
            .0
            .get_or_init(|| {
                let (kernel, concrete) = settle();
                Launcher::new(
                    client,
                    kernel,
                    &concrete,
                    LaunchGrid::Stated {
                        cube_count: self.launch.cube_count.clone(),
                        cube_dim: self.launch.cube_dim,
                    },
                )
                .and_then(|launcher| launcher.holding(self.shared_memory))
                .expect(ADMITTED_PLAN_RUNS)
            })
            .clone()
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
            launcher: SettledLauncher::default(),
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
