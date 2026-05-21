use cubek_std::{CubeDimResource, InvalidConfigError};

use crate::definition::{MatmulSetupError, TilingBlueprint};

use super::{PartitionedStageMatmul, StageMatmul, variant::StageVariant};

/// Marker for the unit-partitioned stage matmul kind: each partition is owned
/// by a single unit.
pub struct UnitPartitioned;

impl StageVariant for UnitPartitioned {
    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        let inner = blueprint.tile_matmul.cubedim_resource()?;
        let factor = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        match inner {
            CubeDimResource::Units(units) => Ok(CubeDimResource::Units(units * factor)),
            _ => Err(Box::new(
                "Error: Tried to use a unit stage matmul with a plane tile matmul.".to_string(),
            )),
        }
    }

    fn validate_blueprint(blueprint: &TilingBlueprint) -> Result<(), MatmulSetupError> {
        let working_units = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        let num_compute_planes =
            Self::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;
        let num_units = blueprint.plane_dim * num_compute_planes;

        if num_units != working_units {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Error: Number of units {num_units} should be {working_units}."
            ))));
        }
        Ok(())
    }

    fn build(data: PartitionedStageMatmul) -> StageMatmul {
        StageMatmul::UnitPartitioned(data)
    }
}
