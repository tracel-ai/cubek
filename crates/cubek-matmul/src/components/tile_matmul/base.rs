use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{InvalidConfigError, TileSize};

use crate::{
    components::{
        resource::CubeDimResource,
        tile_matmul::{Scope, TileConfig},
    },
    definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint},
};

/// A family of [TileMatmul] implementations that operate with any precision.
///
/// There is a single implementor, [TileMatmul](super::TileMatmul),
/// which dispatches on its variant at runtime. The trait still exists to document
/// the surface and to keep the method set grouped, but is expected to be removed
/// once callers are fully migrated to the inherent enum API.
pub trait TileMatmulFamily: Send + Sync + 'static {
    /// Config for this matmul
    type Config: TileConfig;

    /// Compute primitive that executes tile matmuls of this family.
    /// Kept aligned with [cubedim_resource](TileMatmulFamily::cubedim_resource).
    type Scope: Scope;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator(&self) -> bool;

    /// Whether this matmul family is able to cast on load/store from the stage.
    fn can_cast_stage_element(&self) -> bool;

    /// Returns whether this tile matmul may benefit from swizzling.
    /// Used to determine the selection, since swizzling may require different stage sizes.
    fn should_swizzle<R: Runtime>(&self, client: &ComputeClient<R>) -> bool;

    /// Returns the compute resources required to run this matmul.
    fn cubedim_resource(&self) -> Result<CubeDimResource, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and vector sizes.
    fn expand_config(
        &self,
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Returns whether a tile configuration is supported
    fn is_supported<R: Runtime>(&self, client: &ComputeClient<R>, config: MmaConfig) -> bool;

    /// Returns all sizes supported for these types, if any
    fn supported_sizes<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize>;

    fn validate_blueprint<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}
