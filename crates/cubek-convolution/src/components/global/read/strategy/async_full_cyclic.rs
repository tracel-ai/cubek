use std::marker::PhantomData;

use cubecl::{
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
    {ir::DeviceProperties, prelude::barrier::Barrier},
};
use cubek_matmul::components::{
    global::{
        GlobalReaderConfig, PlaneFlowPartition,
        memory::GlobalIterator,
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            FullLoadingStrategy, LoadingJob, LoadingValidation, ReaderMode,
            async_barrier::AsyncCopy,
            async_full_cyclic::AsyncFullCyclicLoading as MatmulCyclicLoading, tiled::TiledLayout,
        },
    },
    stage::{ContiguousTilingLayout, StridedStageFamily, StridedStageMemory, TilingOrder},
};
use cubek_matmul::definition::{MatmulElems, MatmulProblem};
use cubek_std::{InvalidConfigError, StageIdent, tile::Strided};

use crate::components::global::{
    args::RuntimeArgs,
    read::strategy::async_copy::{ASYNC_COPY_WIDTH, async_copy_from},
};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads vectors with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _t: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for AsyncFullCyclicLoading<TO> {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        MatmulCyclicLoading::<TO>::validate_with_config(device_props, config)
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        MatmulCyclicLoading::<TO>::validate_with_problem(problem, dtypes, ident)
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncFullCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        MatmulCyclicLoading::<TO>::max_round_plane_count(
            elements_per_tile,
            tiles_per_stage,
            vector_size,
            plane_dim,
            dtype,
        )
    }
}

#[cube]
impl<TO: TilingOrder> FullLoadingStrategy<RuntimeArgs> for AsyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = AsyncFullCyclicJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        runtime_args: RuntimeArgs,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS> {
        let type_size = ES::type_size_bits().comptime();
        let vector_size = ASYNC_COPY_WIDTH / type_size as u32;
        let tile_num_elements = config.smem_config.elements_per_tile();
        let num_stage_elements = config.smem_config.elements_per_stage();

        let num_stage_vectors = num_stage_elements.div_ceil(vector_size);
        let total_units = config.loading_units_count();
        let num_tasks_per_unit = num_stage_vectors.div_ceil(total_units);
        let balanced_workload = num_stage_vectors.is_multiple_of(total_units);
        let jump_length = total_units * vector_size;

        let unit_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;
        let unit_position_base = unit_id * vector_size;

        AsyncFullCyclicJob {
            unit_position_base,
            runtime_args,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            copy_vector_size: vector_size,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone)]
pub struct AsyncFullCyclicJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    tile_num_elements: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    copy_vector_size: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, TO: TilingOrder>
    LoadingJob<EG, NG, ES, NS, ContiguousTilingLayout<TO>, AsyncCopy> for AsyncFullCyclicJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
        _barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_vector::<EG, NG, ES, NS, TO>(
                this,
                unit_position,
                global_iter,
                stage,
                &this.runtime_args,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                copy_vector::<EG, NG, ES, NS, TO>(
                    this,
                    unit_position,
                    global_iter,
                    stage,
                    &this.runtime_args,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn copy_vector<EG: Numeric, NG: Size, ES: Numeric, NS: Size, TO: TilingOrder>(
    job: &AsyncFullCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
    runtime_args: &RuntimeArgs,
    #[comptime] config: GlobalReaderConfig,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view();

    let tile = ContiguousTilingLayout::<TO>::to_x_y(nth_tile, config.smem_config);

    let pos = layout.to_source_pos((tile, pos_within_tile));
    let stage_offset = unit_position / stage.smem.vector_size() as u32;

    async_copy_from(
        view,
        pos,
        stage,
        stage_offset,
        runtime_args,
        global_iter.offset(),
        config,
        job.copy_vector_size,
    );
}
