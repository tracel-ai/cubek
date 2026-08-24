use crate::multi_level::components::tile::TileMatmulKind;

/// Strategy args that carry a [TileMatmul] kind, so convolution / other crates can
/// construct the strategy with the right tile matmul variant without hardcoding the field name.
pub trait TilingArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind);
}
