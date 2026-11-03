use std::marker::PhantomData;

use cubecl::{
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
    {ir::DeviceProperties, prelude::barrier::Barrier},
};
use cubek_matmul::multi_level::tile::{ContiguousTilingLayout, TilingOrder};
use cubek_matmul::{
    definition::{MatmulElems, MatmulProblem},
    multi_level::{
        components::{
            global::{
                GlobalReaderConfig, PlaneFlowPartition, SharedGlobalMatmulConfig,
                memory::GlobalIterator,
                multi_stage::LoadMaxRoundPlaneCount,
                read::{
                    AsyncPartialLoadingStrategy, LoadingJob, LoadingValidation,
                    PartialLoadingStrategy, ReaderMode, async_barrier::AsyncCopy,
                    async_partial_cyclic::AsyncPartialCyclicLoading as MatmulCyclicLoading,
                    tiled::TiledLayout,
                },
            },
            stage::{StridedStageFamily, StridedStageMemory},
        },
        definition::MatmulTypes,
    },
};
use cubek_std::{InvalidConfigError, StageIdent};

use crate::components::global::{
    args::RuntimeArgs,
    read::strategy::async_copy::{ASYNC_COPY_WIDTH, async_copy_from},
};

#[derive(CubeType, Clone, Copy)]
/// Loads one stage buffer of the tiles using the load-only planes, as the matmul's
/// [`MatmulCyclicLoading`] does, but through the convolution's bounds checks: an im2col position
/// can fall in the padding or past the channels, which the matmul's copy does not check.
pub struct AsyncPartialCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _t: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for AsyncPartialCyclicLoading<TO> {
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

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncPartialCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        dtype: ElemType,
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
impl<TO: TilingOrder> PartialLoadingStrategy<RuntimeArgs> for AsyncPartialCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncCopy;
    type Stage = StridedStageFamily;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = AsyncPartialCyclicJob;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        runtime_args: RuntimeArgs,
        #[comptime] stage_index: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncPartialCyclicJob {
        let type_size = ES::size_bits().comptime();
        let vector_size = ASYNC_COPY_WIDTH / type_size as u32;
        let num_stage_elements = config.smem_config.elements_per_stage();

        let num_vectors_per_tile = config.smem_config.elements_per_tile() / vector_size;
        let total_num_vectors = config.smem_config.tiles_per_stage() * num_vectors_per_tile;
        let total_units = config.loading_units_count();
        let balanced_workload = total_num_vectors.is_multiple_of(total_units);
        let num_tasks_per_unit = total_num_vectors.div_ceil(total_units);
        let jump_length = total_units * vector_size;

        let plane_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow);
        let unit_id = plane_id * config.plane_dim + UNIT_POS_X;
        let unit_position_base = unit_id * vector_size;

        AsyncPartialCyclicJob {
            unit_position_base,
            runtime_args,
            num_tasks_per_unit,
            stage_index,
            jump_length,
            num_vectors_per_tile,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
            copy_vector_size: vector_size,
        }
    }
}

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct AsyncPartialCyclicJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    num_vectors_per_tile: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
    #[cube(comptime)]
    copy_vector_size: u32,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, TO: TilingOrder>
    LoadingJob<EG, NG, ES, NS, ContiguousTilingLayout<TO>, AsyncCopy> for AsyncPartialCyclicJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
        _barrier: &Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;
        let mut stage = stage.with_buffer_index(this.stage_index);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_vector::<EG, NG, ES, NS, TO>(
                &*this,
                unit_position,
                global_iter,
                &mut stage,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                copy_vector::<EG, NG, ES, NS, TO>(
                    &*this,
                    unit_position,
                    global_iter,
                    &mut stage,
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
fn copy_vector<EG: Numeric, NG: Size, ES: Numeric, NS: Size, TO: TilingOrder>(
    job: &AsyncPartialCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view();

    let tile_size = config.smem_config.elements_per_tile();
    let tile_count_row = config.smem_config.tiles_per_stage_along_row();
    let tile_count_col = config.smem_config.tiles_per_stage_along_col();

    let tile_index = unit_position / tile_size;
    let pos_within_tile = unit_position % tile_size;

    let (tile_x_within_stage, tile_y_within_stage) = TO::to_row_col(
        tile_index,
        tile_count_row,
        tile_count_col,
        config.smem_config,
    );

    let tile = match config.stage_ident {
        StageIdent::Lhs => (
            tile_x_within_stage,
            job.stage_index * tile_count_col + tile_y_within_stage,
        ),
        StageIdent::Rhs => (
            job.stage_index * tile_count_row + tile_x_within_stage,
            tile_y_within_stage,
        ),
        _ => unreachable!(),
    };

    let pos = layout.to_source_pos((tile, pos_within_tile));

    let tile_start = tile_index * job.num_vectors_per_tile * job.copy_vector_size;
    let stage_offset = (tile_start + pos_within_tile) / stage.smem.vector_size() as u32;

    async_copy_from(
        view,
        pos,
        stage,
        stage_offset,
        &job.runtime_args,
        global_iter.offset(),
        config,
        job.copy_vector_size,
    );
}

#[cube]
impl<TO: TilingOrder> AsyncPartialLoadingStrategy<RuntimeArgs> for AsyncPartialCyclicLoading<TO> {
    fn arrival_count(#[comptime] config: SharedGlobalMatmulConfig) -> u32 {
        <MatmulCyclicLoading<TO> as AsyncPartialLoadingStrategy<RuntimeArgs>>::arrival_count(config)
    }

    fn barrier_post_init() {
        <MatmulCyclicLoading<TO> as AsyncPartialLoadingStrategy<RuntimeArgs>>::barrier_post_init();
    }

    fn arrive<MP: MatmulTypes>(
        barrier: &Shared<Barrier>,
        #[comptime] config: SharedGlobalMatmulConfig,
    ) {
        <MatmulCyclicLoading<TO> as AsyncPartialLoadingStrategy<RuntimeArgs>>::arrive::<MP>(
            barrier, config,
        );
    }

    fn is_elected(#[comptime] config: SharedGlobalMatmulConfig) -> bool {
        <MatmulCyclicLoading<TO> as AsyncPartialLoadingStrategy<RuntimeArgs>>::is_elected(config)
    }
}
