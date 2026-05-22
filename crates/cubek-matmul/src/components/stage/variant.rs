//! Per-stage-variant contract.

use cubek_std::{CubeDimResource, InvalidConfigError};

use crate::definition::{MatmulSetupError, TilingBlueprint};

use super::{PartitionedStageMatmul, StageMatmul};

pub trait StageVariant: Sized {
    /// Compute resources required for this stage variant on the given
    /// blueprint.
    fn cubedim_resource(blueprint: &TilingBlueprint)
    -> Result<CubeDimResource, InvalidConfigError>;

    /// Per-variant blueprint validation.
    fn validate_blueprint(blueprint: &TilingBlueprint) -> Result<(), MatmulSetupError>;

    /// Wrap shared [`PartitionedStageMatmul`] data into the variant of
    /// [`StageMatmul`] that matches this kind.
    fn build(data: PartitionedStageMatmul) -> StageMatmul;
}
