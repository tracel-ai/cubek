use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use super::block_scaled::{self, QUAD, StagedShape};
use crate::kernels::{
    MatmulBinding, MatmulOutput, MatmulPartitioning, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, cube_order_name,
};
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::CubeOrder;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulRowsKind;
use cubek_microkernel::matmul::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel, CubeStagedMicrokernelSettings,
    CubeStagedStore, MatmulPlaneLayout,
};
use cubek_microkernel::matmul::plane::{PlaneFragmentMicrokernelSettings, PlaneFragmentTransport};
use cubek_std::Hardware;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::microkernel::StageRing;
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` as [`CubeStagedMatmulRoutine`](crate::kernels::CubeStagedMatmulRoutine) runs
/// it, each stage landed by the TMA engine, one descriptor a stage an operand, while every plane
/// contracts ([`CubeStagedDelivery::Tma`]). Its operands are tensor maps, read one element wide;
/// its planes store their own fragments.
///
/// Refused where one box cannot land a stage as its instruction reads it
/// ([`CubeStagedTmaRefusal`](cubek_microkernel::matmul::cube::CubeStagedTmaRefusal)), where the
/// operands are read through an index, and where an input's rows are one element wide.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedTmaMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedTmaMatmulRoutineSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    pub cube_order: CubeOrder,
}

impl CubeStagedTmaMatmulRoutine {
    /// How it fills its stages: the TMA engine, every plane contracting.
    const FILL: CubeStagedFill = CubeStagedFill {
        delivery: CubeStagedDelivery::Tma,
        filling_planes: 0,
    };

    /// The settings a block-scaled product read as stored races on `hw`
    /// (`block_scaled`): for a prompt's rows, four by eight and two by
    /// eight steps on four by two planes and two by four on two by two, two stages in flight; for
    /// a decode's, a stage one step deep, the most one box lands, three or four in flight. None
    /// without a bulk tensor copy.
    pub fn block_scaled_settings(&self, hw: &Hardware) -> Vec<CubeStagedTmaMatmulRoutineSettings> {
        if !hw.bulk_tensor_copy() {
            return Vec::new();
        }
        let shape = |steps, planes, planes_along_k, ring| StagedShape {
            steps,
            planes,
            planes_along_k,
            ring,
        };
        let (double, triple) = (StageRing::DOUBLE, StageRing::TRIPLE);
        // A box lands a stage row of at most a swizzle's 128 bytes, two steps of `K` sharing it
        // at most.
        let shapes = [
            shape((4, 8, 1), (4, 2), 1, double),
            shape((2, 8, 1), (4, 2), 1, double),
            shape((2, 4, 1), (2, 2), 1, double),
            shape((1, 1, 1), (1, 4), 1, QUAD),
            shape((1, 1, 1), (1, 4), 1, triple),
            shape((1, 2, 1), (1, 4), 1, triple),
            shape((1, 2, 1), (1, 2), 2, triple),
            shape((1, 1, 1), (1, 2), 2, QUAD),
        ];
        block_scaled::block_scaled_settings(hw, *self, &shapes, |microkernel| {
            CubeStagedTmaMatmulRoutineSettings {
                microkernel,
                cube_order: CubeOrder::RowMajor,
            }
        })
    }
}

impl core::fmt::Display for CubeStagedTmaMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-tma-matmul {}{}",
            self.microkernel,
            cube_order_name(self.cube_order)
        )
    }
}

impl Routine for CubeStagedTmaMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedTmaMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Through the tile DSL's transport, whose `ldmatrix` reads the rows a box lands swizzled, in
    /// their slots, which is the only ring a bulk copy fills. A ring deeper than two keeps the
    /// engine more than one stage ahead, which a two-stage ring cannot: its next fill waits on
    /// every plane having read the stage before. The boxes are ones that won on the Qwen3
    /// projections at 512 and 2048 rows on a 188-core Blackwell, crossed with the ring each won
    /// with: the `128 × 128` deep box, and the same on two by two planes of `64 × 64`, whose
    /// fragments reuse each operand load twice as much; the `256 × 128` and `128 × 256` boxes, on
    /// eight planes, for the long products. Every one of them fits two stages in 64 KB. None
    /// without a bulk tensor copy.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if !hw.bulk_tensor_copy() {
            return Vec::new();
        }
        let boxed = |(along_m, along_n, depth), (planes_m, planes_n)| {
            CubeStagedMicrokernelSettings::boxed(
                PlaneFragmentMicrokernelSettings::grid(along_m, along_n, depth),
                MatmulPlaneLayout::new(planes_m, planes_n),
            )
        };
        let (double, triple) = (StageRing::DOUBLE, StageRing::TRIPLE);
        [
            ((2, 16, 2), (4, 1), double),
            ((4, 8, 2), (2, 2), triple),
            ((2, 16, 2), (8, 1), double),
            ((2, 16, 2), (4, 2), triple),
        ]
        .map(|(grid, planes, ring)| Self::Settings {
            microkernel: boxed(grid, planes)
                .ringed(ring)
                .through(PlaneFragmentTransport::Mma),
            cube_order: CubeOrder::RowMajor,
        })
        .into()
    }

    fn vectorize(
        &self,
        microkernel: &CubeStagedMicrokernel,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        base::vectorize(microkernel, partitioning, vectorization);
    }

    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &CubeStagedMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        base::estimate(problem, blueprint, grid, estimate);
    }

    /// As [`CubeStagedMatmulRoutine`](crate::kernels::CubeStagedMatmulRoutine)'s, the TMA engine
    /// checked against the stage it lands.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let placement = CubeStagedPlacement::new(
            hw,
            problem,
            &settings.microkernel,
            settings.cube_order,
            Self::FILL,
            CubeStagedStore::Fragments,
        )?;
        placement.check_tma_fill(hw)?;
        placement.prepare(hw)
    }
}

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedTmaMatmulRoutine {
    /// A cube owns a box of the rows.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Tiled;

    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileTensorLaunch, TileTensorLaunch>,
        blueprint: CubeStagedMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        base::dispatch::<O, Out>(tensors, binding, blueprint, grid);
    }
}
