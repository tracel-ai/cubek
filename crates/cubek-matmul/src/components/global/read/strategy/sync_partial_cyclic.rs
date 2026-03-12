use std::marker::PhantomData;

use crate::components::global::read::{PartialLoadingStrategy, tiled::TiledLayout};
use crate::components::global::{GlobalReaderConfig, PlaneFlowPartition};
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::StridedStageMemory;
use crate::components::stage::{ContiguousTilingLayout, TilingOrder};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::definition::{MatmulElems, MatmulProblem, StageIdent};
use crate::{components::global::read::validate_swizzle_atom_size, launch::RuntimeConfig};
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::InvalidConfigError;
use cubek_std::tile::Strided;

use super::{LoadingJob, LoadingValidation, ReaderMode};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads vectors with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct SyncPartialCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for SyncPartialCyclicLoading<TO> {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        if let ReaderMode::Strict = config.reader_mode {
            let vector_size = config.gmem_config.vector_size as u32;
            let num_vectors_per_tile = config.smem_config.elements_per_tile() / vector_size;
            let num_tiles_in_stage = config.smem_config.tiles_per_stage();
            let total_num_vectors = num_tiles_in_stage * num_vectors_per_tile;

            let total_units = config.loading_units_count();
            let jump_length = total_units * vector_size;
            let num_tasks_per_unit = total_num_vectors.div_ceil(total_units);

            let max_id = total_units - 1;
            let max_task_id = num_tasks_per_unit - 1;
            let max_position_base = max_id * vector_size;
            let max_position = max_position_base + max_task_id * jump_length;
            let num_stage_elements = config.smem_config.elements_per_stage();

            if max_position > num_stage_elements {
                return Err(Box::new(
                    "Too many data will be loaded, resulting in out-of-bounds",
                ));
            }
        }

        validate_swizzle_atom_size(config.smem_config)?;
        ContiguousTilingLayout::<TO>::check(config.smem_config)?;

        Ok(())
    }

    fn validate_with_problem(
        _problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        _ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncPartialCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        let num_vectors_per_tile = elements_per_tile / vector_size as u32;
        let total_num_vectors = tiles_per_stage * num_vectors_per_tile;
        total_num_vectors.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder, RC: RuntimeConfig> PartialLoadingStrategy<RC>
    for SyncPartialCyclicLoading<TO>
{
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = SyncPartialCyclicJob;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        _runtime_config: RC,
        #[comptime] stage_index: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> SyncPartialCyclicJob {
        let vector_size = NG::value().comptime() as u32;
        let num_stage_elements = config.smem_config.elements_per_stage();

        let tile_size = config.smem_config.elements_per_tile();
        let tile_count_row = config.smem_config.tiles_per_stage_along_row();
        let tile_count_col = config.smem_config.tiles_per_stage_along_col();

        let num_vectors_per_tile = tile_size / vector_size;
        let total_units = config.loading_units_count();

        let num_tiles_in_stage = tile_count_row * tile_count_col;
        let total_num_vectors = num_tiles_in_stage * num_vectors_per_tile;
        let balanced_workload = total_num_vectors.is_multiple_of(total_units);
        let num_tasks_per_unit = total_num_vectors.div_ceil(total_units);
        let jump_length = total_units * vector_size;

        let plane_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow);
        let unit_id = plane_id * config.plane_dim + UNIT_POS_X;
        let unit_position_base = unit_id * vector_size;

        SyncPartialCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            stage_index,
            jump_length,
            num_vectors_per_tile,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncPartialCyclicJob {
    unit_position_base: u32,

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
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, TO: TilingOrder>
    LoadingJob<EG, NG, ES, NS, ContiguousTilingLayout<TO>, Synchronous> for SyncPartialCyclicJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
        _barrier: &mut (),
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;
        let mut stage = stage.with_buffer_index(this.stage_index);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            load_and_store_vector::<EG, NG, ES, NS, TO>(
                this,
                unit_position,
                global_iter,
                &mut stage,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                load_and_store_vector::<EG, NG, ES, NS, TO>(
                    this,
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
pub(crate) fn load_and_store_vector<
    EG: Numeric,
    NG: Size,
    ES: Numeric,
    NS: Size,
    TO: TilingOrder,
>(
    job: &SyncPartialCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view().view(layout);

    let tile_size = config.smem_config.elements_per_tile();
    let tile_count_row = config.smem_config.tiles_per_stage_along_row();
    let tile_count_col = config.smem_config.tiles_per_stage_along_col();
    let vector_size = view.vector_size();

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

    let vector_read = view.read_checked((tile, pos_within_tile));

    let tile_start = tile_index * job.num_vectors_per_tile;
    let mut tile_slice = stage.as_slice_mut::<NS>();
    let offset = tile_start + pos_within_tile / vector_size as u32;
    let type_size = Vector::<ES, NS>::type_size();
    let offset = stage.swizzle.apply(offset, type_size);

    tile_slice[offset as usize] = Vector::cast_from(vector_read);
}
