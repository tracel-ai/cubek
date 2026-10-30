//! The matmul microkernels on one work item: `out = lhs · rhs` over a problem one tile wide.

use crate::matmul::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings, plane_fragment_staged_microkernel,
};
use crate::matmul::{
    MatmulAxes, MatmulDestination, MatmulLayout, MatmulOperands, MatmulReading, MatmulRows,
    MatmulStage, MatmulWorkItem,
};
use cubecl::prelude::*;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{
    Launcher, Partitioning, Scratch, Semiring, Space, StageStorage, TileArg, TileArgLaunch,
    TileSpec,
};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{Microkernel, StageRing};

/// The elements of a float product: what both inputs are stored in, what it sums in and what it
/// writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulElements {
    pub input: ElemType,
    pub sum: ElemType,
    pub out: ElemType,
}

impl MatmulElements {
    /// A float product of row-major inputs into a buffer, its tiles cut by the launch.
    pub fn work_item(&self) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: MatmulOperands {
                lhs: self.input,
                rhs: self.input,
                out: self.out,
            },
            sum: self.sum,
            layouts: MatmulOperands::all(MatmulLayout::Row),
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::MNK,
            reading: MatmulReading::AsStored,
        }
    }
}

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
            &stage.operands(item.elements, microkernel.orders),
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

    /// `out = lhs · rhs`, `lhs` row-major `m × k`, `rhs` row-major `k × n` and `out` `m × n`.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(
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
        let MatmulElements {
            input,
            sum,
            out: out_element,
        } = self.elements;
        let bound = |arg, element, axes: &[_]| {
            TileArgLaunch::new(
                arg,
                element,
                1,
                TileSpec::direct(axes).boundary(BoundaryPolicy::Every(Boundary::Zero)),
            )
        };
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            plane_fragment_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                bound(lhs, input, &[M, K]),
                bound(rhs, input, &[K, N]),
                bound(out, out_element, &[M, N]),
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
#[allow(clippy::too_many_arguments)]
fn plane_fragment_kernel<EA: Numeric, EL: Numeric>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &TileArg<'_>,
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
