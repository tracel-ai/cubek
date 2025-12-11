mod t1x1x1 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 1, n: 1, k: 1 })
    }

    include!("partition.rs");
}

mod t8x1x4 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 8, n: 1, k: 4 })
    }

    include!("partition.rs");
}

mod t2x4x1 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 2, n: 4, k: 1 })
    }

    include!("partition.rs");
}

mod t1x8x8 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 1, n: 8, k: 8 })
    }

    include!("partition.rs");
}

mod t4x4x4 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 4, n: 4, k: 4 })
    }

    include!("partition.rs");
}

mod t8x8x8 {
    use super::*;
    use cubek_matmul::components::{TileSize, TilingScheme, TilingSchemeBuilder};

    fn tile_size(builder: TilingSchemeBuilder) -> TilingSchemeBuilder {
        builder.with_tile_size(TileSize { m: 8, n: 8, k: 8 })
    }

    include!("partition.rs");
}
