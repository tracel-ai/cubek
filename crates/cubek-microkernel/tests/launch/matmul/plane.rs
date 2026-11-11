//! The plane matmul microkernels on one work item.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{ComputeScope, CubeOrder, PlaneOrder};
use cubecl::tile::{
    Instruction, Launcher, Partitioning, Region, Scratch, Semiring, Space, StageStorage, Stages,
    Tile, TileTensor,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::matmul::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings, PlaneShuffleMicrokernel,
    PlaneShuffleMicrokernelSettings, plane_fragment_microkernel, plane_fragment_partial,
    plane_shuffle_microkernel,
};
use cubek_microkernel::matmul::{
    MatmulAxes, MatmulLayout, MatmulMicrokernel, MatmulOperands, MatmulRead, MatmulRows,
    MatmulStage,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{Microkernel, StageRing};

use super::{MatmulElements, MatmulLaunch, OneCube, WholeRows};

/// One plane fragment microkernel ready to launch on one work item: the microkernel the device
/// picks at `settings`, and the problem one plane's tile wide, `steps` stages deep.
#[derive(Debug, Clone)]
pub struct PlaneFragmentLaunch {
    pub microkernel: PlaneFragmentMicrokernel,
    pub elements: MatmulElements,
    /// `m`, `n` and `k` of the problem.
    pub shape: [usize; 3],
    stage: StageStorage,
    ring: StageRing,
    plane_width: u32,
}

impl PlaneFragmentLaunch {
    /// The best plane fragment microkernel `hw` runs at `settings`, over a problem `steps` stages
    /// of `K` deep.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &PlaneFragmentMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item();
        let microkernel = PlaneFragmentMicrokernel::best(hw, &item, settings)?;
        let [m, n, k] = [M, N, K].map(|axis| microkernel.tile_extent(axis));
        let stage = MatmulStage::new(Space::new(&[(M, m), (N, n), (K, k)]), MatmulAxes::MNK);
        let plane_width = microkernel.instruction.units(hw)?;
        let ring = StageRing::SINGLE.resolve(
            hw,
            &stage.operands(
                item.elements,
                microkernel.orders,
                MatmulOperands::all(MatmulRead::AsBound),
            ),
            plane_width as usize,
        );
        let storage = microkernel.instruction.stage_layout(None)?.storage(
            stage.space(),
            microkernel.instruction.stage_tile(
                &[],
                MatmulAxes::MNK,
                microkernel.settings.rows_along,
            ),
        );
        Ok(Self {
            microkernel,
            elements,
            shape: [m, n, k * steps],
            stage: storage,
            ring,
            plane_width,
        })
    }
}

impl MatmulLaunch for PlaneFragmentLaunch {
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
        let [m, n, k] = self.shape;
        let space = Space::new(&[(M, m), (N, n), (K, k)]);
        let partitioning = Partitioning::new(
            space.clone(),
            self.microkernel
                .tiling()
                .walk_every(&[K])
                .launch_planes(
                    &[M, N],
                    PlaneOrder::new(&[(M, 1), (N, 1)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let MatmulElements { input, sum, .. } = self.elements;
        let [lhs, rhs, out] = self.elements.bound([lhs, rhs, out], self.rhs_layout());
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            plane_fragment_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                lhs,
                rhs,
                out,
                launcher.partitioning_arg(),
                self.microkernel,
                self.stage.clone(),
                self.ring,
                [sum, input],
            );
        }
        Ok(())
    }
}

/// The one plane walking its tile through its own stages
/// ([`plane_fragment_staged_microkernel`]) and draining its fragments into `out`.
#[cube(launch_unchecked)]
fn plane_fragment_kernel<EA: Numeric, EL: Numeric>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] stage: StageStorage,
    #[comptime] ring: StageRing,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);

    for plane in &partitioning {
        let (c, a, b) = (c.at(&plane), a.at(&plane), b.at(&plane));

        let mut sum = c
            .accumulator::<EA>(&a, &b, microkernel.tile_instruction(), Semiring::SUM_PROD)
            .with_scratch(Scratch::None);
        sum.zero();

        plane_fragment_staged_microkernel::<EL>(
            &plane,
            &sum,
            &c,
            &a,
            &b,
            microkernel,
            stage.clone(),
            ring,
        );

        sum.drained_into(&c);
    }
}

/// One plane's tile walked through stages of its own: the plane stages its operands a step of its
/// walk at a time, holding as many stages as `ring` says, and contracts every stage into `sum`
/// with the plane microkernel. `sum`, `c`, `a` and `b` are windowed to the plane; a narrow step's
/// partial is opened over `c`. A walk below the planes' level is the plane's own, so its stages
/// are too: each plane fills a copy of its own, met on its own barriers.
#[cube]
fn plane_fragment_staged_microkernel<EL: Numeric>(
    plane: &Region,
    sum: &Tile,
    c: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] stage: StageStorage,
    #[comptime] ring: StageRing,
) {
    let instruction = microkernel.tile_instruction();
    let step_sum = microkernel.instruction.step_sum();
    let partial = plane_fragment_partial::<EL>(c, a, b, step_sum, instruction);

    let walk = plane.walk();
    let in_flight = ring.stages_in_flight.get();
    let mut stages = Stages::smem(&walk, a, b, stage, in_flight, None);

    stages.walk(walk, ring.prefetch, |slot, step| {
        let sum_step = sum.at(step);
        slot.consume(|a_step, b_step| {
            plane_fragment_microkernel(
                step,
                &sum_step,
                &partial,
                a_step,
                b_step,
                instruction,
                None,
                MatmulOperands::AS_BOUND,
            );
        });
    });
}

/// One plane shuffle microkernel ready to launch on one plane's box, every row held whole: the
/// microkernel the device picks at `settings`, `steps` of its reads of `K` deep.
#[derive(Debug, Clone)]
pub struct PlaneShuffleLaunch {
    pub microkernel: PlaneShuffleMicrokernel,
    pub elements: MatmulElements,
    one_cube: OneCube,
}

impl PlaneShuffleLaunch {
    /// A plane shuffle reads a row rhs.
    const RHS: MatmulLayout = MatmulLayout::Row;

    /// The plane shuffle microkernel `hw` runs at `settings` over `rows`.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &PlaneShuffleMicrokernelSettings,
        rows: WholeRows,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item_of(MatmulRows::Whole(rows.0), Self::RHS);
        let microkernel = PlaneShuffleMicrokernel::best(hw, &item, settings)?;
        let levels = microkernel
            .tiling()
            .launch_planes(&[M, N], PlaneOrder::new(&[(N, 1)], CubeOrder::RowMajor))
            .build();
        Ok(Self {
            microkernel,
            elements,
            one_cube: OneCube::new(levels, ComputeScope::Plane, steps),
        })
    }
}

impl MatmulLaunch for PlaneShuffleLaunch {
    fn elements(&self) -> MatmulElements {
        self.elements
    }

    fn shape(&self) -> [usize; 3] {
        self.one_cube.shape
    }

    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let launcher = self
            .one_cube
            .launcher(client, self.microkernel.plane_width() as u32)?;
        let [lhs, rhs, out] = self.elements.bound([lhs, rhs, out], self.rhs_layout());
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; it holds no shared memory.
        unsafe {
            plane_shuffle_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                lhs,
                rhs,
                out,
                launcher.partitioning_arg(),
                self.microkernel.tile_instruction(),
                [self.elements.sum, self.elements.out],
            );
        }
        Ok(())
    }
}

/// The one plane walking `K` through its own reads, summing in registers and draining into `out`.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
fn plane_shuffle_kernel<EA: Numeric, E: Numeric>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] instruction: Instruction,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);
    for plane in &partitioning {
        let (c, a, b) = (c.at(&plane), a.at(&plane), b.at(&plane));
        let mut sum = c.accumulator::<EA>(&a, &b, instruction, Semiring::SUM_PROD);
        sum.zero();
        for k_step in plane.walk_within(&b) {
            plane_shuffle_microkernel(&k_step, &sum, &a, &b, MatmulOperands::AS_BOUND);
        }
        sum.drained_into(&c);
    }
}
