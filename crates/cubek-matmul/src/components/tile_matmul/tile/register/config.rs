use cubek_std::{MatrixLayout, TileSize};

use crate::definition::SwizzleModes;

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

impl ProductType {
    pub(crate) fn from_layouts(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        tile_size: TileSize,
    ) -> Self {
        let lhs_preferred = match lhs_layout {
            MatrixLayout::RowMajor => ProductType::Inner,
            MatrixLayout::ColMajor => ProductType::Outer,
        };
        let rhs_preferred = match rhs_layout {
            MatrixLayout::RowMajor => ProductType::Outer,
            MatrixLayout::ColMajor => ProductType::Inner,
        };

        if lhs_preferred == rhs_preferred {
            lhs_preferred
        } else if tile_size.m() == 1 {
            rhs_preferred
        } else if tile_size.n() == 1 {
            lhs_preferred
        } else {
            // No better solution
            ProductType::Outer
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterMatmulConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub product_type: ProductType,
}

impl RegisterMatmulConfig {
    pub fn new(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        tile_size: TileSize,
        plane_dim: u32,
        swizzle_modes: SwizzleModes,
    ) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
            product_type: ProductType::from_layouts(lhs_layout, rhs_layout, tile_size),
        }
    }
}
