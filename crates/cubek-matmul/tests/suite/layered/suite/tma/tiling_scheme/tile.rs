mod t16x16x16 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize {
            m: 16,
            n: 16,
            k: 16,
        })
    }

    include!("partition.rs");
}

#[cfg(feature = "matmul_tests_mma")]
mod t16x8x16 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 16, n: 8, k: 16 })
    }

    include!("partition.rs");
}
