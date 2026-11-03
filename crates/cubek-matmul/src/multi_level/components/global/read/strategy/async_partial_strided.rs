use crate::{
    definition::{MatmulElems, MatmulProblem, StageIdent},
    multi_level::{
        args::RuntimeConfig,
        components::{
            global::{
                GlobalReaderConfig, PlaneFlowPartition, SharedGlobalMatmulConfig,
                memory::GlobalIterator,
                multi_stage::LoadMaxRoundPlaneCount,
                read::{
                    AsyncPartialLoadingStrategy, PartialLoadingStrategy,
                    async_barrier::AsyncCopy,
                    async_copy::{ASYNC_COPY_WIDTH, async_copy_from},
                    stage::FullStageLayout,
                    validate_async_barrier, validate_async_copy, validate_async_copy_with_problem,
                    validate_swizzle_atom_size,
                },
            },
            stage::{StridedStageFamily, StridedStageMemory},
        },
        definition::MatmulTypes,
        tile::{StridedTilingLayout, TilingValidation},
    },
};
use cubecl::{
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
    {ir::DeviceProperties, prelude::barrier::Barrier},
};
use cubek_std::InvalidConfigError;

use super::{LoadingJob, LoadingValidation, ReaderMode, unevenly_split_stage};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads vectors with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncPartialStridedLoading {}

impl LoadingValidation for AsyncPartialStridedLoading {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        let vector_size = ASYNC_COPY_WIDTH / config.smem_config.dtype.size_bits() as u32;

        // Needs separate check because copy size may be larger than global vector size
        if !config
            .smem_config
            .elements_per_stage_along_contiguous_dim()
            .is_multiple_of(vector_size)
        {
            return Err(Box::new("Stage size isn't divisible by copy vector size"));
        }

        if let ReaderMode::Strict = config.reader_mode {
            let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size;
            let total_units = config.loading_units_count();

            if !num_stage_vectors.is_multiple_of(total_units) {
                return Err(unevenly_split_stage(num_stage_vectors, total_units));
            }
        }

        validate_swizzle_atom_size(config.smem_config)?;
        validate_async_barrier(device_props)?;
        validate_async_copy(
            device_props,
            &config.gmem_config.dtype,
            &config.smem_config.dtype,
        )?;
        StridedTilingLayout::check(config.smem_config)?;

        Ok(())
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        validate_async_copy_with_problem(problem, dtypes, ident)
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        _vector_size: VectorSize,
        plane_dim: u32,
        dtype: ElemType,
    ) -> u32 {
        let vector_size = ASYNC_COPY_WIDTH / dtype.size_bits() as u32;
        let num_vectors_per_tile = elements_per_tile / vector_size;
        let total_num_vectors = tiles_per_stage * num_vectors_per_tile;
        total_num_vectors.div_ceil(plane_dim)
    }
}

#[cube]
impl<RC: RuntimeConfig> PartialLoadingStrategy<RC> for AsyncPartialStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = AsyncPartialStridedJob;
    type Stage = StridedStageFamily;
    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        _runtime_config: RC,
        #[comptime] stage_index: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS> {
        let type_size = ES::size_bits().comptime();
        let vector_size = ASYNC_COPY_WIDTH / type_size as u32;

        let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size;
        let unit_count = config.loading_planes_count() * config.plane_dim;
        let num_tasks_per_unit = num_stage_vectors.div_ceil(unit_count);
        let balanced_workload = num_stage_vectors.is_multiple_of(unit_count);

        let unit_position_base = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        AsyncPartialStridedJob {
            stage_index,
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            copy_vector_size: vector_size,
            balanced_workload,
            num_stage_vectors,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone))]
pub struct AsyncPartialStridedJob {
    unit_position_base: u32,

    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    copy_vector_size: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_vectors: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size>
    LoadingJob<EG, NG, ES, NS, StridedTilingLayout, AsyncCopy> for AsyncPartialStridedJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, StridedTilingLayout>,
        _barrier: &Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_vector::<EG, NG, ES, NS>(&*this, unit_position, global_iter, stage, config);
        } else {
            if unit_position < this.num_stage_vectors {
                copy_vector::<EG, NG, ES, NS>(&*this, unit_position, global_iter, stage, config);
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
fn copy_vector<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
    job: &AsyncPartialStridedJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut StridedStageMemory<ES, NS, StridedTilingLayout>,
    #[comptime] config: GlobalReaderConfig,
) {
    let mut stage = stage.with_buffer_index(job.stage_index);

    let unit_position_abs = unit_position * job.copy_vector_size;

    let layout = FullStageLayout::new(config.smem_config);
    let view = global_iter.view();

    let pos = layout.to_source_pos(unit_position_abs);
    let pos = match config.stage_ident {
        StageIdent::Lhs => (
            pos.0,
            pos.1 + config.smem_config.elements_per_stage_along_col() * job.stage_index,
        ),
        StageIdent::Rhs => (
            pos.0 + config.smem_config.elements_per_stage_along_row() * job.stage_index,
            pos.1,
        ),
        _ => pos,
    };

    let stage_offset = unit_position_abs / stage.smem.vector_size() as u32;

    async_copy_from(
        view,
        pos,
        &mut stage,
        stage_offset,
        config,
        job.copy_vector_size,
    );
}

#[cube]
impl<RC: RuntimeConfig> AsyncPartialLoadingStrategy<RC> for AsyncPartialStridedLoading {
    fn arrival_count(#[comptime] config: SharedGlobalMatmulConfig) -> u32 {
        let total_load_units = config.plane_flow_config().counts.load_only * config.plane_dim();
        total_load_units.runtime()
    }

    fn barrier_post_init() {}

    fn arrive<MP: MatmulTypes>(
        barrier: &Shared<Barrier>,
        #[comptime] _config: SharedGlobalMatmulConfig,
    ) {
        barrier.commit_copy_async();
        barrier.arrive();
    }

    fn is_elected(#[comptime] config: SharedGlobalMatmulConfig) -> bool {
        let role_rule = PlaneFlowPartition::new(config.plane_flow_config().partition_rule);
        role_rule.is_load_plane()
    }
}
