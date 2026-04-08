use std::marker::PhantomData;

use crate::{
    components::global::read::{FullLoadingStrategy, tiled::TiledLayout},
    components::global::{GlobalReaderConfig, PlaneFlowPartition},
    components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous},
    components::stage::StridedStageFamily,
    components::stage::{ContiguousTilingLayout, StridedStageMemory, TilingOrder},
    components::{global::memory::GlobalIterator, stage::TilingValidation},
    definition::{MatmulElems, MatmulProblem, StageIdent},
    {components::global::read::validate_swizzle_atom_size, launch::RuntimeConfig},
};
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{InvalidConfigError, tile::Strided};

use super::{LoadingJob, LoadingValidation, ReaderMode};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads vectors with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct SyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _t: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for SyncFullCyclicLoading<TO> {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        if let ReaderMode::Strict = config.reader_mode {
            let vector_size = config.gmem_config.vector_size;

            let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size as u32;
            let total_units = config.loading_units_count();

            if !num_stage_vectors.is_multiple_of(total_units) {
                return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting vector size and number of planes so that total unit count {total_units:?} divides number of vectors in stage.",
            )));
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

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncFullCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        let elements_per_stage = elements_per_tile * tiles_per_stage;
        let num_vectors = elements_per_stage / vector_size as u32;
        num_vectors.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder, RC: RuntimeConfig> FullLoadingStrategy<RC> for SyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = SyncFullCyclicJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        _runtime_config: RC,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS> {
        let vector_size = NG::value().comptime() as u32;
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

        SyncFullCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullCyclicJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    tile_num_elements: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, TO: TilingOrder>
    LoadingJob<EG, NG, ES, NS, ContiguousTilingLayout<TO>, Synchronous> for SyncFullCyclicJob
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

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            load_and_store_vector::<EG, NG, ES, NS, TO>(
                this,
                unit_position,
                global_iter,
                stage,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                load_and_store_vector::<EG, NG, ES, NS, TO>(
                    this,
                    unit_position,
                    global_iter,
                    stage,
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
    job: &SyncFullCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<TO>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view().view(layout);

    let tile = ContiguousTilingLayout::<TO>::to_x_y(nth_tile, config.smem_config);

    let mut slice = stage.as_slice_mut::<NS>();

    let vector_read = view.read_checked((tile, pos_within_tile));
    let stage_offs = stage.swizzle.apply(unit_position, ES::type_size());

    slice[stage_offs as usize / NS::value()] = Vector::cast_from(vector_read);
}
