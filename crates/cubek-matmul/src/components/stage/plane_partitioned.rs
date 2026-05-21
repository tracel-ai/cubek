use cubek_std::{CubeDimResource, InvalidConfigError};

use crate::definition::{MatmulSetupError, TilingBlueprint};

use super::{PartitionedStageMatmul, StageMatmul, variant::StageVariant};

/// Marker for the plane-partitioned stage matmul kind: each partition is
/// owned by a full plane.
pub struct PlanePartitioned;

impl StageVariant for PlanePartitioned {
    fn cubedim_resource(
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        let inner = blueprint.tile_matmul.cubedim_resource()?;
        let factor = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        match inner {
            CubeDimResource::Planes(planes) => Ok(CubeDimResource::Planes(planes * factor)),
            _ => Err(Box::new(
                "Error: Tried to use a plane stage matmul with a unit tile matmul.".to_string(),
            )),
        }
    }

    fn validate_blueprint(blueprint: &TilingBlueprint) -> Result<(), MatmulSetupError> {
        let num_planes_needed = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        let num_compute_planes =
            Self::cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;

        if num_compute_planes != num_planes_needed {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Error: Number of compute planes {num_compute_planes} should be {num_planes_needed}."
            ))));
        }
        Ok(())
    }

    fn build(data: PartitionedStageMatmul) -> StageMatmul {
        StageMatmul::PlanePartitioned(data)
    }
}
