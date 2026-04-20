use cubek_std::{TileSize, stage::SwizzleMode, tile::mma::MmaIOConfig};

use crate::{definition::StageIdent, definition::SwizzleModes};
use std::{fmt::Debug, hash::Hash};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TileKind {
    Cmma,
    Mma,
    Register,
    PlaneVec,
    Interleaved,
}

/// Execution mode for the RegisterMatmul
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ProductType {
    /// Computes the Tile Matmul as m*n inner products of length k.
    ///
    /// Needs Lhs to be row major and Rhs to be col major
    /// If not the case, tile will be transposed during load
    Inner,
    /// Computes the Stage Matmul as the sum of k outer products of size m*n.
    ///
    /// Needs Lhs to be col major and Rhs to be row major
    /// If not the case, tile will be transposed during load
    Outer,
}

// This serves as interface for higher level matmuls, not for what is used within tile matmul
pub trait TileConfig: Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static {
    fn kind(&self) -> TileKind;

    /// Returns the vector size for the given ident
    fn plane_dim(&self) -> u32;

    fn elements_in_tile_m(&self) -> u32;

    fn elements_in_tile_n(&self) -> u32;

    fn elements_in_tile_k(&self) -> u32;

    /// Returns the [SwizzleMode] for the given ident
    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode;

    // --- Specialized getters with defaults ---

    fn mma_io_config(&self) -> MmaIOConfig {
        panic!("MmaIOConfig not available for this tile config")
    }

    fn product_type(&self) -> ProductType {
        ProductType::Inner
    }

    fn reduce_vector_size(&self) -> u32 {
        1
    }
}

/// Configuration for the Tile Matmul level
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SharedTileConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl TileConfig for SharedTileConfig {
    fn kind(&self) -> TileKind {
        TileKind::Cmma
    }

    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn elements_in_tile_m(&self) -> u32 {
        self.tile_size.m()
    }

    fn elements_in_tile_n(&self) -> u32 {
        self.tile_size.n()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.tile_size.k()
    }

    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        match ident {
            StageIdent::Lhs => self.swizzle_modes.lhs,
            StageIdent::Rhs => self.swizzle_modes.rhs,
            StageIdent::Acc => self.swizzle_modes.acc,
            StageIdent::Out => self.swizzle_modes.out,
        }
    }
}

impl SharedTileConfig {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle: SwizzleModes) -> Self {
        SharedTileConfig {
            tile_size,
            plane_dim,
            swizzle_modes: swizzle,
        }
    }

    // Direct accessors — used by per-kind layout types in tile.
    // These duplicate TileConfig trait methods so layouts don't depend on the trait.

    pub fn tile_m(&self) -> u32 {
        self.tile_size.m()
    }

    pub fn tile_n(&self) -> u32 {
        self.tile_size.n()
    }

    pub fn tile_k(&self) -> u32 {
        self.tile_size.k()
    }
}
