//! The hypercube matmul microkernel on one box whose `K` its cubes share: each cube walks its run
//! of the box's stages into one sum, and the box's last arrival adds the runs' parts up.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Grid, LastArrival, Refusal};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{
    Launcher, Partitioning, Space, StageStorage, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::matmul::cube::CubeStagedWalk;
use cubek_microkernel::matmul::hypercube::{
    HypercubeStagedMicrokernel, HypercubeStagedMicrokernelSettings,
    hypercube_staged_last_arrival_microkernel,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::Microkernel;

use super::{MatmulElements, MatmulLaunch, OneCube};

/// The cubes sharing the box's `K`: three, so a box three stages deep is cut into three parts the
/// last arrival adds up, and one a stage deep is held whole by the first run.
const CUBES: usize = 3;

/// One hypercube staged microkernel ready to launch on one box, `steps` stages of `K` deep, its
/// `K` shared by [`CUBES`] cubes.
#[derive(Debug, Clone)]
pub struct HypercubeStagedLaunch {
    pub microkernel: HypercubeStagedMicrokernel,
    pub elements: MatmulElements,
    storage: StageStorage,
    plane_width: u32,
    shape: [usize; 3],
    partitioning: Partitioning,
}

impl HypercubeStagedLaunch {
    /// # Errors
    ///
    /// Why the device runs none: no cube staged microkernel at `settings`, stages past its shared
    /// memory, or cubes whose stores do not reach one another within a launch.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &HypercubeStagedMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        if !hw.properties.features.device_memory_scope {
            return Err(Refusal::Declined {
                why: "the last arrival hands one cube's stores to another within a launch",
            });
        }
        let item = elements.work_item();
        let microkernel = HypercubeStagedMicrokernel::best(hw, &item, settings)?;
        if microkernel.cube.walk().decoded().is_some() {
            return Err(Refusal::Declined {
                why: "a cube that decodes its stages binds words, which this launch's plain bindings do not",
            });
        }
        let placed = OneCube::new(
            microkernel.cube.tiling().cubes(&[M, N]).build(),
            ComputeScope::Cube,
            steps,
        );
        let resolved = microkernel.cube.resolve(
            hw,
            &microkernel.cube.contracted(&item),
            &placed.cube_space(),
        )?;
        if resolved.shared_memory > hw.shared_memory_budget() {
            return Err(Refusal::Declined {
                why: "the cube's stages hold more shared memory than the device has",
            });
        }
        let microkernel = HypercubeStagedMicrokernel {
            cube: resolved.microkernel,
        };
        let one_cube = OneCube::new(
            microkernel.cube.launch_tiling().cubes(&[M, N]).build(),
            ComputeScope::Cube,
            steps,
        );
        let [m, n, k] = one_cube.shape;
        let partitioning = Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            microkernel
                .cube
                .launch_tiling()
                .cubes(&[M, N])
                .shared_by(CUBES)
                .build(),
        );
        Ok(Self {
            microkernel,
            elements,
            storage: resolved.storage,
            plane_width: resolved.plane_width,
            shape: one_cube.shape,
            partitioning,
        })
    }
}

impl MatmulLaunch for HypercubeStagedLaunch {
    fn elements(&self) -> MatmulElements {
        self.elements
    }

    fn shape(&self) -> [usize; 3] {
        self.shape
    }

    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let partitioning = &self.partitioning;
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        let launcher = Launcher::new(client, partitioning.clone(), partitioning.space(), grid)?;
        let [lhs, rhs, out] = self.elements.bound([lhs, rhs, out], self.rhs_layout());
        let slots = TensorHandle::empty(
            client,
            partitioning.arrival_slots(&[M, N]),
            self.elements.sum,
        );
        let slots = TileTensorLaunch::new(
            slots.into_arg(),
            self.elements.sum,
            1,
            TileSpec::direct(&[M, N]),
        );
        let arrivals = TensorHandle::zeros(
            client,
            [partitioning.arrival_counters()],
            u32::elem_type_native(),
        );
        // SAFETY: every operand is bound through the launcher at the problem's own extents, the
        // kernel indexes no buffer but through its tiles and the arrival counter of the box, which
        // arrives holding zero; its shared memory is its stages.
        unsafe {
            hypercube_staged_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                1,
                lhs,
                rhs,
                out,
                slots,
                arrivals.into_arg().into_buffer_arg(),
                launcher.partitioning_arg(),
                self.microkernel.cube.walk(),
                self.storage.clone(),
                [self.elements.sum, self.elements.input],
            );
        }
        Ok(())
    }
}

/// Every cube walking its run of the box's stages, handing its sum on through the box's last
/// arrival.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn hypercube_staged_kernel<EA: Numeric, EL: Numeric, VO: Size>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    slots: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);
    let stages = comptime!(partitioning.levels()[1].clone());
    let portion = partitioning.walk().portion(stages.clone());
    for i in 0..portion.touched() {
        let tile = portion.region(i);
        let (from, steps) = portion.steps(i);
        let merge: LastArrival<'_> = slots.last_arrival::<VO>(&partitioning, &portion, i, arrivals);
        hypercube_staged_last_arrival_microkernel::<EA, EL>(
            tile.over(&stages).range(from, steps),
            &a.at(&tile),
            &b.at(&tile),
            merge,
            &c,
            walk,
            storage.clone(),
        );
    }
}
