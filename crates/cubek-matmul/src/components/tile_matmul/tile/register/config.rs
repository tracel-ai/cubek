use cubek_std::{MatrixLayout, stage::SwizzleMode};

use crate::components::tile_matmul::{ProductType, SharedTileConfig, TileConfig};

use crate::definition::StageIdent;

impl ProductType {
    pub(crate) fn from_layouts(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        config: &SharedTileConfig,
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
        } else if config.tile_size.m() == 1 {
            rhs_preferred
        } else if config.tile_size.n() == 1 {
            lhs_preferred
        } else {
            // No better solution
            ProductType::Outer
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterMatmulConfig {
    pub shared: SharedTileConfig,
    pub product_type: ProductType,
}

impl RegisterMatmulConfig {
    pub fn from_shared_tile_config(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        config: SharedTileConfig,
    ) -> Self {
        Self {
            shared: config,
            product_type: ProductType::from_layouts(lhs_layout, rhs_layout, &config),
        }
    }
}

impl TileConfig for RegisterMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim()
    }

    fn elements_in_tile_m(&self) -> u32 {
        self.shared.elements_in_tile_m()
    }

    fn elements_in_tile_n(&self) -> u32 {
        self.shared.elements_in_tile_n()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.shared.elements_in_tile_k()
    }

    fn swizzle_mode(&self, ident: StageIdent) -> SwizzleMode {
        self.shared.swizzle_mode(ident)
    }

    fn product_type(&self) -> ProductType {
        self.product_type
    }
}
