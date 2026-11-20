use crate::multi_level::{components::tile::TileMatmulKind, routines::PlaneTilingBlueprintOptions};

/// Strategy args that carry a `TileMatmul` kind, so convolution / other crates can
/// construct the strategy with the right tile matmul variant without hardcoding the field name.
pub trait TilingArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind);
    /// Hand the routine's plane selector these options in place of its own; a no-op for a routine
    /// without a plane selector.
    fn set_plane_tiling(&mut self, _options: PlaneTilingBlueprintOptions) {}
}
