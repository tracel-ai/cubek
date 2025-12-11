mod t1x8x256 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 1, n: 8, k: 256 })
    }

    include!("partition.rs");
}

mod t1x4x128 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 1, n: 4, k: 128 })
    }

    include!("partition.rs");
}

mod t1x1x128 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 1, n: 1, k: 128 })
    }

    include!("partition.rs");
}
