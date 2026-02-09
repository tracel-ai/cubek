use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_matmul::{
    components::{
        global::{
            GlobalReaderConfig, PlaneFlowPartition,
            memory::GlobalIterator,
            multi_stage::LoadMaxRoundPlaneCount,
            read::{
                FullLoadingStrategy, LoadingJob, LoadingValidation, sync::Synchronous,
                validate_swizzle_atom_size,
            },
        },
        stage::{NoTilingLayout, TilingValidation},
        tile::io::Strided,
    },
    definition::{InvalidConfigError, MatmulElems, MatmulProblem, StageIdent},
    launch::RuntimeConfig,
};

use crate::components::stage::{
    bias_stage::{BiasStageFamily, BiasStageMemory},
    reader::BiasTilingLayout,
};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct SyncBiasLoading {}

impl LoadingValidation for SyncBiasLoading {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        validate_swizzle_atom_size(config.smem_config)?;
        BiasTilingLayout::check(config.smem_config)?;

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

impl LoadMaxRoundPlaneCount for SyncBiasLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: LineSize,
        plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        let elements_per_stage = elements_per_tile * tiles_per_stage;
        let num_lines = elements_per_stage / line_size as u32;
        num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<RC: RuntimeConfig> FullLoadingStrategy<RC> for SyncBiasLoading {
    type TilingLayout = NoTilingLayout;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = SyncBiasJob;
    type Stage = BiasStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, ES: Numeric>(
        _runtime_config: RC,
        #[comptime] line_size: LineSize,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let num_stage_elements = config.smem_config.elements_per_stage_along_contiguous_dim();

        let num_stage_lines = num_stage_elements.div_ceil(line_size as u32);
        let total_units = config.loading_units_count();
        let num_tasks_per_unit = num_stage_lines.div_ceil(total_units);
        let balanced_workload = num_stage_lines.is_multiple_of(total_units);
        let jump_length = total_units * line_size as u32;

        let unit_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;
        let unit_position_base = unit_id * line_size as u32;

        SyncBiasJob {
            unit_position_base,
            num_tasks_per_unit,
            jump_length,
            line_size,
            balanced_workload,
            num_stage_elements,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncBiasJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    line_size: LineSize,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, NoTilingLayout, Synchronous> for SyncBiasJob {
    type Stage = BiasStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut BiasStageMemory<ES>,
        _barrier: &mut (),
        #[comptime] _config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.balanced_workload) {
            load_and_store_line::<EG, ES>(this, unit_position, global_iter, stage);
        } else {
            if unit_position < this.num_stage_elements {
                load_and_store_line::<EG, ES>(this, unit_position, global_iter, stage);
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn load_and_store_line<EG: Numeric, ES: Numeric>(
    job: &SyncBiasJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut BiasStageMemory<ES>,
) {
    let line_size = job.line_size;

    let view = global_iter.view();

    let mut slice = stage.as_slice_mut(line_size);

    let line_read = view.read_checked((0, unit_position));
    let stage_offs = stage.swizzle.apply(unit_position, ES::type_size());

    slice[stage_offs as usize / job.line_size] = Line::cast_from(line_read);
}
