//! Per-tile-variant contract.
//!
//! Each `*Matmul` config struct (`CmmaMatmul`, `MmaMatmul`, …) implements
//! [`TileVariant`] so the [`TileMatmulKind`](super::TileMatmulKind) enum can
//! dispatch to per-variant logic via a uniform match-and-call without owning
//! any variant-specific logic itself. The methods are associated functions
//! (no `&self`): [`TileMatmulKind`](super::TileMatmulKind) is the unit-only
//! selector consulted *before* an instance is built (e.g. `expand`,
//! `validate_blueprint`, `cubedim_resource`).

use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{CubeDimResource, InvalidConfigError, TileSize};

use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint};

pub trait TileVariant: Sized {
    /// Whether this tile matmul requires specialized hardware accelerators
    /// (e.g. tensor cores).
    fn requires_accelerator() -> bool;

    /// Whether this kind supports a cast on load/store from the stage.
    fn can_cast_stage_element() -> bool;

    /// Whether this kind benefits from swizzling on the given client.
    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool;

    /// Compute resources required to run this kind.
    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError>;

    /// Whether a specific MMA configuration is supported on the given client.
    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool;

    /// All sizes supported for the given element-type triple, if any.
    fn supported_sizes<R: Runtime>(
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize>;

    /// Build the per-kind config from the matmul-flow inputs.
    fn expand(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Self;

    /// Per-kind blueprint validation.
    fn validate<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}
