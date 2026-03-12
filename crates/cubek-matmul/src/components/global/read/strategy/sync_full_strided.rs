use crate::components::global::read::{FullLoadingStrategy, stage::FullStageLayout};
use crate::components::global::{GlobalReaderConfig, PlaneFlowPartition};
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{StridedStageMemory, StridedTilingLayout};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::definition::{MatmulElems, MatmulProblem, StageIdent};
use crate::{components::global::read::validate_swizzle_atom_size, launch::RuntimeConfig};
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::InvalidConfigError;
use cubek_std::tile::Strided;

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the stage using all planes,
/// keeping the original layout, making each tile strided
pub struct SyncFullStridedLoading {}

impl LoadingValidation for SyncFullStridedLoading {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        let vector_size = config.gmem_config.vector_size;

        let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size as u32;
        let total_units = config.loading_units_count();

        if !num_stage_vectors.is_multiple_of(total_units) {
            return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting vector size and number of planes so that total unit count {total_units:?} divides number of vectors in stage.",
            )));
        }

        validate_swizzle_atom_size(config.smem_config)?;
        StridedTilingLayout::check(config.smem_config)?;

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

impl LoadMaxRoundPlaneCount for SyncFullStridedLoading {
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
impl<RC: RuntimeConfig> FullLoadingStrategy<RC> for SyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = SyncFullStridedJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        _runtime_config: RC,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS> {
        let vector_size = NG::value().comptime() as u32;
        let num_stage_vectors = config.smem_config.elements_per_stage() / vector_size;
        let unit_count = config.loading_planes_count() * config.plane_dim;
        let num_tasks_per_unit = num_stage_vectors / unit_count;

        let unit_position_base = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        SyncFullStridedJob {
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullStridedJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size>
    LoadingJob<EG, NG, ES, NS, StridedTilingLayout, Synchronous> for SyncFullStridedJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, StridedTilingLayout>,
        _barrier: &mut (),
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        let layout = FullStageLayout::new(config.smem_config);
        let view = global_iter.view().view(layout);

        let vector_read = view.read_checked(unit_position * NG::value() as u32);
        let type_size = Vector::<ES, NS>::type_size();
        let stage_offs = stage.swizzle.apply(unit_position, type_size);

        stage.as_slice_mut::<NS>()[stage_offs as usize] = Vector::cast_from(vector_read);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
