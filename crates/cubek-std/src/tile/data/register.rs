use cubecl::prelude::*;

use crate::{
    MatrixLayout, SwizzleModes, TileSize,
    tile::{
        LOGIT_MASKED, Tile, TileKind, TileScope,
        compute::{Mask, MaskExpand},
        data::rowwise::RowWise,
    },
};

#[derive(CubeType)]
pub struct RegisterTile<N: Numeric> {
    pub data: Array<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: RegisterMatmul,
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

impl ProductType {
    pub fn from_layouts(
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
pub struct RegisterMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub product_type: ProductType,
}

impl RegisterMatmul {
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

#[cube]
impl<E: Float> RegisterTile<E> {
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        acc.copy_from(base);
        for r in 0..m as usize {
            let row_offset = r as u32 * n;
            let mut val = E::min_value();
            for c in 0..n {
                val = max(val, self.data[(row_offset + c) as usize]);
            }
            acc.vals[r] = max(acc.vals[r], val);
        }
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        acc.fill(E::from_int(0));
        for r in 0..m as usize {
            let row_offset = r as u32 * n;
            let mut val = E::from_int(0);
            for c in 0..n {
                val += self.data[(row_offset + c) as usize];
            }
            acc.vals[r] += val;
        }
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        let threshold = E::new(LOGIT_MASKED);
        for r in 0..m as usize {
            let row_offset = r as u32 * n;
            let val = rowwise.vals[r];
            let safe_val = clamp_min(val, threshold);
            let not_masked = E::cast_from(val >= threshold);
            for c in 0..n {
                let idx = (row_offset + c) as usize;
                self.data[idx] = not_masked * (self.data[idx] - safe_val).exp();
            }
        }
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        for r in 0..m as usize {
            let row_offset = r as u32 * n;
            for c in 0..n {
                let idx = (row_offset + c) as usize;
                self.data[idx] = self.data[idx] * scale.vals[r];
            }
        }
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        for r in 0..m {
            let row_offset = r * n;
            for c in 0..n {
                let idx = (row_offset + c) as usize;
                self.data[idx] = self.data[idx] * scale
                    + E::cast_from(mask.should_mask((r, c))) * E::min_value();
            }
        }
    }

    pub fn fill_zero(&mut self) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        for i in 0..m * n {
            self.data[i as usize] = E::from_int(0);
        }
    }

    /// Cast-copies this register tile into `dest`. Used by per-variant softmax
    /// helpers when writing the post-softmax score into a same-storage
    /// destination.
    pub fn write_to<Lhs: Float>(&self, dest: &mut RegisterTile<Lhs>) {
        let m = comptime!(self.config.tile_size.m());
        let n = comptime!(self.config.tile_size.n());
        for i in 0..m * n {
            dest.data[i as usize] = Lhs::cast_from(self.data[i as usize]);
        }
    }
}

#[cube]
pub fn register_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
) -> Tile<L, Sc, ReadWrite> {
    Tile::from_kind(TileKind::new_Register(RegisterTile::<L> {
        data: Array::new((config.tile_size.m() * config.tile_size.k()) as usize),
        matrix_layout: layout,
        config,
    }))
}

#[cube]
pub fn register_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
) -> Tile<R, Sc, ReadWrite> {
    Tile::from_kind(TileKind::new_Register(RegisterTile::<R> {
        data: Array::new((config.tile_size.n() * config.tile_size.k()) as usize),
        matrix_layout: layout,
        config,
    }))
}

#[cube]
pub fn register_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
) -> Tile<A, Sc, ReadWrite> {
    Tile::from_kind(TileKind::new_Register(RegisterTile::<A> {
        data: Array::new((config.tile_size.m() * config.tile_size.n()) as usize),
        matrix_layout: layout,
        config,
    }))
}
