use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::ContiguousTilingLayout;
use crate::components::stage::OrderedTilingOrder;
use crate::components::{global::PlaneFlowPartition, stage::TilingValidation};
use crate::components::{global::read::FullLoadingStrategy, stage::StridedStageFamily};
use crate::components::{global::read::validate_swizzle_atom_size, tile::io::Strided};
use crate::definition::FormattedConfigError;
use crate::definition::InvalidConfigError;
use crate::definition::MatmulElems;
use crate::definition::MatmulProblem;
use crate::definition::StageIdent;
use crate::{components::global::GlobalReaderConfig, launch::RuntimeConfig};
use cubecl::{ir::DeviceProperties, prelude::*};

use super::{LoadingValidation, sync_full_tilewise};

#[derive(CubeType, Clone, Copy)]
/// Similar to `sync_full_tilewise`, but includes additional validation checks.
///
/// This function operates only on the LHS (left-hand side).
///
/// - In the single-row case, behavior is similar to `tilewise` with row-major tiling order.
///   However, it will explicitly fail if any plane does not load its entire row.
/// - In the multi-row case, it too will fail if a plane does not load all its rows.
///   Within each plane, the local tiling order is column-major.
pub struct SyncFullOrderedLoading {}

impl LoadingValidation for SyncFullOrderedLoading {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        if config.stage_ident != StageIdent::Lhs {
            return Err(FormattedConfigError::new(move || {
                "Ordered loading only available on Lhs".to_string()
            }));
        }

        let line_size = config.gmem_config.line_size;
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();

        if !num_tiles.is_multiple_of(num_planes) {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for ordered loading.",
                )
            }));
        }

        let num_tiles_per_plane = num_tiles / num_planes;
        let num_lines_per_tile = config.smem_config.elements_per_tile() / line_size as u32;
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_planes = config.loading_planes_count();
        let plane_dim = config.plane_dim;
        let rows_per_plane = config.smem_config.tiles_per_stage_along_row() / num_planes;

        if !num_lines_per_plane.is_multiple_of(plane_dim) {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Plane dimension {plane_dim:?} must divide number of lines per plane {num_lines_per_plane:?} for ordered loading.",
                )
            }));
        }

        let tile_count_col = config.smem_config.tiles_per_stage_along_col();
        if num_tiles_per_plane != rows_per_plane * tile_count_col {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of tiles per plane {num_tiles_per_plane:?} must equal rows_per_plane {rows_per_plane:?} times cols {tile_count_col:?} for ordered loading.",
                )
            }));
        }

        validate_swizzle_atom_size(config.smem_config)?;
        ContiguousTilingLayout::<OrderedTilingOrder>::check(config.smem_config)?;

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

impl LoadMaxRoundPlaneCount for SyncFullOrderedLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: LineSize,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        tiles_per_stage
    }
}

#[cube]
impl<RC: RuntimeConfig> FullLoadingStrategy<RC> for SyncFullOrderedLoading {
    type TilingLayout = ContiguousTilingLayout<OrderedTilingOrder>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = sync_full_tilewise::SyncFullTilewiseJob;
    type Stage = StridedStageFamily;
    type TileKind = Strided;

    fn new_job<EG: Numeric, ES: Numeric>(
        _runtime_config: RC,
        #[comptime] line_size: LineSize,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();
        let plane_dim = config.plane_dim;

        let num_tiles_per_plane = num_tiles / num_planes;
        let num_lines_per_tile = config.smem_config.elements_per_tile() / line_size as u32;
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let num_tiles_to_skip = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * num_tiles_per_plane;
        let num_lines_to_skip = num_tiles_to_skip * num_lines_per_tile;

        // Ordered is just a tilewise reader using the ordered tiling order
        sync_full_tilewise::SyncFullTilewiseJob {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim,
            line_size,
        }
    }
}
