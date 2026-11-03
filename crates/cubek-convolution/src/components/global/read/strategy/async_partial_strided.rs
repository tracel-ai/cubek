use cubecl::{
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
    {ir::DeviceProperties, prelude::barrier::Barrier},
};
use cubek_matmul::multi_level::tile::StridedTilingLayout;
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
                    async_partial_strided::AsyncPartialStridedLoading as MatmulStridedLoading,
                    stage::FullStageLayout,
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
/// Loads one stage buffer in its original layout using the load-only planes, as the matmul's
/// [`MatmulStridedLoading`] does, but through the convolution's bounds checks: an im2col position
/// can fall in the padding or past the channels, which the matmul's copy does not check.
pub struct AsyncPartialStridedLoading {}

impl LoadingValidation for AsyncPartialStridedLoading {
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        MatmulStridedLoading::validate_with_config(device_props, config)
    }

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        MatmulStridedLoading::validate_with_problem(problem, dtypes, ident)
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        dtype: ElemType,
    ) -> u32 {
        MatmulStridedLoading::max_round_plane_count(
            elements_per_tile,
            tiles_per_stage,
            vector_size,
            plane_dim,
            dtype,
        )
    }
}

#[cube]
impl PartialLoadingStrategy<RuntimeArgs> for AsyncPartialStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = AsyncPartialStridedJob;
    type Stage = StridedStageFamily;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        runtime_args: RuntimeArgs,
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
            runtime_args,
            num_tasks_per_unit,
            unit_count,
            copy_vector_size: vector_size,
            balanced_workload,
            num_stage_vectors,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct AsyncPartialStridedJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

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
        &job.runtime_args,
        global_iter.offset(),
        config,
        job.copy_vector_size,
    );
}

#[cube]
impl AsyncPartialLoadingStrategy<RuntimeArgs> for AsyncPartialStridedLoading {
    fn arrival_count(#[comptime] config: SharedGlobalMatmulConfig) -> u32 {
        <MatmulStridedLoading as AsyncPartialLoadingStrategy<RuntimeArgs>>::arrival_count(config)
    }

    fn barrier_post_init() {
        <MatmulStridedLoading as AsyncPartialLoadingStrategy<RuntimeArgs>>::barrier_post_init();
    }

    fn arrive<MP: MatmulTypes>(
        barrier: &Shared<Barrier>,
        #[comptime] config: SharedGlobalMatmulConfig,
    ) {
        <MatmulStridedLoading as AsyncPartialLoadingStrategy<RuntimeArgs>>::arrive::<MP>(
            barrier, config,
        );
    }

    fn is_elected(#[comptime] config: SharedGlobalMatmulConfig) -> bool {
        <MatmulStridedLoading as AsyncPartialLoadingStrategy<RuntimeArgs>>::is_elected(config)
    }
}
