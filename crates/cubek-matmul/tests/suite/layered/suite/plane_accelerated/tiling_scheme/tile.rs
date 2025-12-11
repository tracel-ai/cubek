#[cfg(target_os = "macos")]
mod t8x8x8 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 8, n: 8, k: 8 })
    }

    include!("partition.rs");
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
mod t32x8x16 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 32, n: 8, k: 16 })
    }

    include!("partition.rs");
}

#[cfg(not(target_os = "macos"))]
mod t8x32x16 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 8, n: 32, k: 16 })
    }

    include!("partition.rs");
}

#[cfg(not(target_os = "macos"))]
mod t16x16x8 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 16, n: 16, k: 8 })
    }

    include!("partition.rs");
}

#[cfg(feature = "matmul_tests_mma")]
mod t16x8x16 {
    use super::*;

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 16, n: 8, k: 16 })
    }

    include!("partition.rs");
}
