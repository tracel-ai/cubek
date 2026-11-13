use core::num::NonZeroUsize;

use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use super::block_scaled::{self, StagedShape};
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
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings;
use cubek_std::Hardware;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::microkernel::StageRing;
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` with the TMA engine landing each stage
/// ([`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine)), or planes set
/// aside issuing it ([`CubeStagedTmaFillerMatmulRoutine`](crate::kernels::CubeStagedTmaFillerMatmulRoutine)),
/// and the box stored by the TMA engine too: the planes drain it into shared memory laid dense,
/// and one bulk copy stores it, clipped at the output's edge, every row of the box leaving whole
/// ([`CubeStagedStore::Tma`]).
///
/// Refused where its fill is, and where a descriptor cannot store the box
/// ([`CubeStagedTmaStoreRefusal`](cubek_microkernel::matmul::cube::CubeStagedTmaStoreRefusal)): an
/// output a sink or one whose rows are one element wide, among others.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedTmaStoredMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedTmaStoredMatmulRoutineSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    /// Planes set aside to fill the stages, beside the ones that contract: none, or as many as
    /// [`CubeStagedTmaFillerMatmulRoutine`](crate::kernels::CubeStagedTmaFillerMatmulRoutine) sets
    /// aside.
    pub filling_planes: usize,
    pub cube_order: CubeOrder,
}

impl CubeStagedTmaStoredMatmulRoutine {
    /// The settings a block-scaled product read as stored races on `hw`
    /// (`block_scaled`): for a prompt's rows, four by four steps on two by
    /// four planes, and two by four on four by two or two by four planes, three stages in flight,
    /// which a contraction of a few thousand finishes sooner with; and the first beside two
    /// filling planes, four stages in flight. None without a bulk tensor copy.
    pub fn block_scaled_settings(
        &self,
        hw: &Hardware,
    ) -> Vec<CubeStagedTmaStoredMatmulRoutineSettings> {
        if !hw.bulk_tensor_copy() {
            return Vec::new();
        }
        let shape = |steps, planes| StagedShape {
            steps,
            planes,
            planes_along_k: 1,
            ring: StageRing::TRIPLE,
        };
        let shapes = [
            shape((4, 4, 1), (2, 4)),
            shape((2, 4, 1), (4, 2)),
            shape((2, 4, 1), (2, 4)),
        ];
        let settings = |filling_planes| {
            move |microkernel| CubeStagedTmaStoredMatmulRoutineSettings {
                microkernel,
                filling_planes,
                cube_order: CubeOrder::RowMajor,
            }
        };
        let mut listed = block_scaled::block_scaled_settings(hw, *self, &shapes, settings(0));
        // The `128 × 128` box beside two filling planes: the activation's scale words, which the
        // units copy, are then theirs alone, so the planes that contract never wait for a slot to
        // empty before filling it. The box takes the ring's bytes once the walk is over, which
        // leaves room for a fourth stage. On an RTX PRO 6000, Devstral Small 2's gate/up at 512
        // rows: 313 µs every plane filling, 297 beside two filling planes over three stages, 281
        // over four.
        let deep = StagedShape {
            ring: StageRing {
                stages_in_flight: NonZeroUsize::new(4).unwrap(),
                ..StageRing::TRIPLE
            },
            ..shapes[0]
        };
        listed.extend(block_scaled::block_scaled_settings(
            hw,
            *self,
            &[deep],
            settings(2),
        ));
        listed
    }
}

impl core::fmt::Display for CubeStagedTmaStoredMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-tma-stored-matmul {} filling_planes={}{}",
            self.microkernel,
            self.filling_planes,
            cube_order_name(self.cube_order)
        )
    }
}

impl Routine for CubeStagedTmaStoredMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedTmaStoredMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// From the same race as [`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine)'s:
    /// the `64 × 128` box four steps deep beside a filling plane, whose stored box costs no cube a
    /// core where the stages already hold one alone; and the `64 × 64` box beside a filling plane,
    /// which a projection with few columns needs to fill the device at all: a 512-row chunk of
    /// Qwen3-0.6B's output projection is 128 of the `64 × 128` boxes for 188 cores, and on the
    /// `64 × 64` it takes 16.4 µs against 22.5, what cuBLAS takes. Both fit their ring and box in
    /// 99 KB. None without a bulk tensor copy.
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
        [((2, 8, 4), (2, 2)), ((2, 4, 4), (2, 2))]
            .map(|(grid, planes)| Self::Settings {
                microkernel: boxed(grid, planes).ringed(StageRing::TRIPLE),
                filling_planes: 1,
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

    /// As [`CubeStagedTmaMatmulRoutine`](crate::kernels::CubeStagedTmaMatmulRoutine)'s, the TMA
    /// engine checked against the box it stores too, before the box is sized
    /// (`CubeStagedPlacement::new`).
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let fill = CubeStagedFill {
            delivery: CubeStagedDelivery::Tma,
            filling_planes: settings.filling_planes,
        };
        let placement = CubeStagedPlacement::new(
            hw,
            problem,
            &settings.microkernel,
            settings.cube_order,
            fill,
            CubeStagedStore::Tma,
        )?;
        placement.check_tma_fill(hw)?;
        placement.prepare(hw)
    }
}

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedTmaStoredMatmulRoutine {
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
