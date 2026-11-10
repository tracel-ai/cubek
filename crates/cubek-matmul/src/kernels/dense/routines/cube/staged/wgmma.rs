use super::base::{self, CubeStagedMatmulRoutineBlueprint, CubeStagedPlacement};
use crate::kernels::{
    MatmulBinding, MatmulOutput, MatmulPartitioning, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, cube_order_name,
};
use cubecl::tile::launch::{DestinationLaunch, Refusal, Vectorization};
use cubecl::tile::space::CubeOrder;
use cubecl::tile::{StageStorage, TileTensorLaunch};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::MatmulRowsKind;
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
use cubek_microkernel::matmul::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernel, CubeStagedMicrokernelSettings,
    CubeStagedStore,
};
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings;
use cubek_std::Hardware;
use cubek_std::microkernel::KernelEstimate;
use cubek_std::microkernel::StageRing;
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` on a warpgroup MMA: groups of planes issue each instruction together out of
/// stages the TMA engine lands swizzled, issued by planes set aside to fill, which release a stage
/// only once the MMAs reading it are done; each group's sum drains straight into its output.
///
/// Refused where the device or product takes no warpgroup MMA, where the TMA engine cannot land
/// the stages, and where the sum would be merged, along `K` or through a bounce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedWgmmaMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedWgmmaMatmulRoutineSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    pub cube_order: CubeOrder,
}

impl CubeStagedWgmmaMatmulRoutine {
    /// How it fills its stages: the TMA engine, issued by one plane set aside, which a warpgroup
    /// MMA needs to release a stage only once its MMAs are done with it.
    const FILL: CubeStagedFill = CubeStagedFill {
        delivery: CubeStagedDelivery::Tma,
        filling_planes: 1,
    };
}

impl core::fmt::Display for CubeStagedWgmmaMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-wgmma-matmul {}{}",
            self.microkernel,
            cube_order_name(self.cube_order)
        )
    }
}

impl Routine for CubeStagedWgmmaMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = CubeStagedMicrokernel;
    type Settings = CubeStagedWgmmaMatmulRoutineSettings;
    type Blueprint = CubeStagedMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Two plane groups along `m`, each one `64 × 128` instruction four steps of `K` a stage (a
    /// 128-byte swizzle span of `f16`), two stages in flight. None without a bulk tensor copy and
    /// a warpgroup MMA.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if !hw.bulk_tensor_copy() || hw.properties.features.matmul.wgmma.iter().next().is_none() {
            return Vec::new();
        }
        vec![Self::Settings {
            microkernel: CubeStagedMicrokernelSettings::boxed(
                PlaneFragmentMicrokernelSettings::grid(1, 16, 4),
                MatmulPlaneLayout::new(2, 1),
            )
            .ringed(StageRing::DOUBLE)
            .on_warpgroup(),
            cube_order: CubeOrder::RowMajor,
        }]
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

    /// As [`CubeStagedTmaFillerMatmulRoutine`](crate::kernels::CubeStagedTmaFillerMatmulRoutine)'s
    /// over one filling plane; the microkernel refuses a warpgroup MMA its fill and drain do not
    /// serve.
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

impl MatmulRoutine<CubeStagedMicrokernel, StageStorage> for CubeStagedWgmmaMatmulRoutine {
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
