use crate::components::global::read::{validate_async_barrier, validate_tma};
use crate::components::global::{PlaneFlowPartition, read::async_tma::AsyncTma};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{StridedStageMemory, SwizzleMode};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{global::multi_stage::LoadMaxRoundPlaneCount, stage::TmaTilingLayout};
use crate::components::{
    global::read::{FullLoadingStrategy, validate_tma_with_problem},
    tile::io::Strided,
};
use crate::definition::{InvalidConfigError, MatmulElems, MatmulProblem, MatrixLayout, StageIdent};
use crate::{components::global::GlobalReaderConfig, launch::RuntimeConfig};
use cubecl::prelude::*;
use cubecl::{ir::DeviceProperties, prelude::barrier::Barrier};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using TMA load instructions.
/// Uses special tiling to minimize the number of loads required. Issues one load for each
/// tile in the major dimension (i.e. `k` for col-major RHS).
pub struct AsyncFullTmaLoading {}

impl LoadingValidation for AsyncFullTmaLoading {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        TmaTilingLayout::check(config.smem_config)?;
        validate_async_barrier(device_props)?;
        validate_tma(device_props, &config.smem_config, &config.gmem_config.dtype)?;

        Ok(())
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        validate_tma_with_problem(problem, dtypes, ident)
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullTmaLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        _tiles_per_stage: u32,
        _line_size: LineSize,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        // Not sure this is the best value, but TMA is executed per-warpgroup so this is the maximum
        // number of planes executing one set of TMA loads.
        4
    }
}

#[cube]
impl<RC: RuntimeConfig> FullLoadingStrategy<RC> for AsyncFullTmaLoading {
    type TilingLayout = TmaTilingLayout;
    type SyncStrategy = AsyncTma;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullTmaJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, ES: Numeric>(
        _runtime_config: RC,
        #[comptime] _line_size: LineSize,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let role_rule_config = config.plane_flow_config.partition_rule;
        let config = config.smem_config;
        let tile_count_col = match config.matrix_layout {
            MatrixLayout::RowMajor => config.tiles_per_stage_along_col(),
            MatrixLayout::ColMajor => config.tiles_per_stage_along_row(),
        };
        // Swizzle renders the column format irrelevant, so we load the whole stage at once
        // The tiling is set on launch for TMA, so no further change is needed here.
        let num_tasks = match config.swizzle {
            SwizzleMode::None => tile_count_col,
            _ => 1u32,
        };

        let is_elected = PlaneFlowPartition::new(role_rule_config).elect_load_leader();

        AsyncFullTmaJob {
            is_elected,
            num_tasks,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullTmaJob {
    is_elected: bool,

    #[cube(comptime)]
    num_tasks: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, TmaTilingLayout, AsyncTma> for AsyncFullTmaJob {
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, TmaTilingLayout>,
        barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        if this.is_elected {
            let config = config.smem_config;

            let size_row = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_per_stage_along_row(),
                MatrixLayout::ColMajor => config.elements_per_stage_along_col(),
            };
            let size_col = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_per_tile_along_col,
                MatrixLayout::ColMajor => config.elements_per_tile_along_row,
            };

            let global_view = global_iter.view();
            let mut stage = stage.as_slice_mut(stage.smem.line_size());
            let slice_size = size_row * size_col / stage.line_size() as u32;

            let slice_start = task_id * slice_size;
            let slice = stage.slice_mut(slice_start as usize, (slice_start + slice_size) as usize);
            let col = task_id * size_col;

            let pos = match config.matrix_layout {
                MatrixLayout::RowMajor => (0u32, col).runtime(),
                MatrixLayout::ColMajor => (col, 0u32).runtime(),
            };

            global_view.tensor_map_load(barrier, &mut slice.downcast(), pos);
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
