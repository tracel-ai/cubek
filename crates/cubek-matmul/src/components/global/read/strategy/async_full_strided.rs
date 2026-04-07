use crate::components::global::read::{
    FullLoadingStrategy, stage::FullStageLayout, validate_async_barrier,
};
use crate::{
    components::global::read::async_copy::ASYNC_COPY_WIDTH,
    components::global::read::validate_async_copy_with_problem,
};
use crate::{
    components::global::read::{async_copy::async_copy_from, validate_swizzle_atom_size},
    components::global::{GlobalReaderConfig, PlaneFlowPartition},
    components::global::{multi_stage::LoadMaxRoundPlaneCount, read::validate_async_copy},
    components::stage::StridedStageFamily,
    components::stage::{StridedStageMemory, StridedTilingLayout},
    components::{global::memory::GlobalIterator, stage::TilingValidation},
    definition::{MatmulElems, MatmulProblem, StageIdent},
    {components::global::read::async_barrier::AsyncCopy, launch::RuntimeConfig},
};
use cubecl::{
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
    {ir::DeviceProperties, prelude::barrier::Barrier},
};
use cubek_std::{InvalidConfigError, tile::Strided};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the stage using all planes,
/// keeping the original layout, making each tile strided
pub struct AsyncFullStridedLoading {}

impl LoadingValidation for AsyncFullStridedLoading {
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

        let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size;
        let total_units = config.loading_units_count();

        if !num_stage_vectors.is_multiple_of(total_units) {
            return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting vector size and number of planes so that total unit count {total_units:?} divides number of vectors in stage.",
            )));
        }

        validate_async_barrier(device_props)?;
        validate_swizzle_atom_size(config.smem_config)?;
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

impl LoadMaxRoundPlaneCount for AsyncFullStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        _vector_size: VectorSize,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        let vector_size = ASYNC_COPY_WIDTH / dtype.size_bits() as u32;
        let elements_per_stage = elements_per_tile * tiles_per_stage;
        let num_vectors = elements_per_stage / vector_size;
        num_vectors.div_ceil(plane_dim)
    }
}

#[cube]
impl<RC: RuntimeConfig> FullLoadingStrategy<RC> for AsyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = AsyncFullStridedJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        _runtime_args: RC,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS> {
        let type_size = ES::type_size_bits().comptime();
        let vector_size = ASYNC_COPY_WIDTH / type_size as u32;
        let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size;
        let unit_count = config.loading_planes_count() * config.plane_dim;
        let num_tasks_per_unit = num_stage_vectors / unit_count;

        let unit_position_base = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        AsyncFullStridedJob {
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            copy_vector_size: vector_size,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullStridedJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    copy_vector_size: u32,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size>
    LoadingJob<EG, NG, ES, NS, StridedTilingLayout, AsyncCopy> for AsyncFullStridedJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, StridedTilingLayout>,
        _barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;
        let unit_position_abs = unit_position * this.copy_vector_size;

        let layout = FullStageLayout::new(config.smem_config);
        let view = global_iter.view();

        let pos = layout.to_source_pos(unit_position_abs);
        let stage_offset = unit_position_abs / stage.smem.vector_size() as u32;

        async_copy_from(
            view,
            pos,
            stage,
            stage_offset,
            config,
            this.copy_vector_size,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
