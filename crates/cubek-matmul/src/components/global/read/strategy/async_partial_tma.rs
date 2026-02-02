use crate::components::global::read::{validate_async_barrier, validate_tma};
use crate::components::global::{GlobalConfig, GlobalReaderConfig};
use crate::components::global::{PlaneFlowPartition, multi_stage::LoadMaxRoundPlaneCount};
use crate::components::stage::TmaTilingLayout;
use crate::components::stage::{StridedStageMemory, SwizzleMode};
use crate::components::{
    global::SharedGlobalMatmulConfig,
    stage::{StageConfig, StridedStageFamily},
};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{
    global::read::{PartialLoadingStrategy, async_tma::AsyncTma},
    tile::io::Strided,
};
use crate::definition::{
    InvalidConfigError, LhsS, MatmulElems, MatmulPrecision, MatmulProblem, MatrixLayout, RhsS,
    StageIdent,
};
use crate::{
    components::global::read::{AsyncPartialLoadingStrategy, validate_tma_with_problem},
    launch::RuntimeConfig,
};
use cubecl::prelude::*;
use cubecl::{ir::DeviceProperties, prelude::barrier::Barrier};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using TMA load instructions.
/// Uses special tiling to minimize the number of loads required. Issues one load for each
/// tile in the major dimension (i.e. `k` for col-major RHS).
pub struct AsyncPartialTmaLoading {}

impl LoadingValidation for AsyncPartialTmaLoading {
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

impl LoadMaxRoundPlaneCount for AsyncPartialTmaLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        _tiles_per_stage: u32,
        _line_size: LineSize,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        4
    }
}

#[cube]
impl<RC: RuntimeConfig> PartialLoadingStrategy<RC> for AsyncPartialTmaLoading {
    type TilingLayout = TmaTilingLayout;
    type SyncStrategy = AsyncTma;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    type Job<EG: Numeric, ES: Numeric> = AsyncPartialTmaJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        _runtime_config: RC,
        #[comptime] stage_index: u32,
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

        AsyncPartialTmaJob {
            is_elected,
            num_tasks,
            stage_index,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialTmaJob {
    is_elected: bool,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    stage_index: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, TmaTilingLayout, AsyncTma>
    for AsyncPartialTmaJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, TmaTilingLayout>,
        barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        if this.is_elected {
            let size_row = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => config.smem_config.elements_per_stage_along_row(),
                MatrixLayout::ColMajor => config.smem_config.elements_per_stage_along_col(),
            };
            let size_col = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => config.smem_config.elements_per_tile_along_col,
                MatrixLayout::ColMajor => config.smem_config.elements_per_tile_along_row,
            };

            let (offs_row, offs_col) = comptime![match config.stage_ident {
                StageIdent::Lhs => (
                    0,
                    this.stage_index * config.smem_config.elements_per_stage_along_col()
                ),
                StageIdent::Rhs => (
                    this.stage_index * config.smem_config.elements_per_stage_along_row(),
                    0
                ),
                _ => (0, 0),
            }]
            .runtime();

            let global_view = global_iter.view();
            let mut stage = stage.as_slice_mut(1usize);
            let slice_size = size_row * size_col;

            let slice_start = task_id * slice_size;
            let slice = stage.slice_mut(slice_start as usize, (slice_start + slice_size) as usize);
            // "column" to be loaded, may be a row for col-major (can't think of a better name)
            let load_col = task_id * size_col;

            let pos = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => (offs_row, load_col + offs_col),
                MatrixLayout::ColMajor => (load_col + offs_row, offs_col),
            };

            global_view.tensor_map_load(barrier, &mut slice.downcast(), pos);
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}

#[cube]
impl<RC: RuntimeConfig> AsyncPartialLoadingStrategy<RC> for AsyncPartialTmaLoading {
    fn arrival_count<S: StageConfig>(#[comptime] _config: SharedGlobalMatmulConfig<S>) -> u32 {
        1u32.runtime()
    }

    fn barrier_post_init() {
        sync_async_proxy_shared();
    }

    fn arrive<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    ) {
        let lhs_elem_size = LhsS::<MP>::type_size().comptime();
        let rhs_elem_size = RhsS::<MP>::type_size().comptime();
        let lhs_bytes =
            config.lhs_reader_config().smem_config.elements_per_stage() * lhs_elem_size as u32;
        let rhs_bytes =
            config.rhs_reader_config().smem_config.elements_per_stage() * rhs_elem_size as u32;
        let stage_bytes = lhs_bytes + rhs_bytes;
        barrier.arrive_and_expect_tx(1, stage_bytes);
    }

    fn is_elected<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> bool {
        let role_rule = PlaneFlowPartition::new(config.plane_flow_config().partition_rule);
        role_rule.elect_load_leader()
    }
}
