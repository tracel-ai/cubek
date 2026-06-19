//! The flat 1-D view over a [`Tile`]. [`FlatLayout`] is a [`Layout`] that re-views the tile's N-D
//! [`Space`] as a single row-major [`Coords1d`] index (`shape()` is the element count);
//! [`Tile::flat`]/[`Tile::flat_mut`] then wrap it as a [`FlatView`]/[`FlatViewMut`] (a
//! [`MaskedView`] carrying the comptime overhang-`check` flag). Used by elementwise leaves such as
//! dequantize, which scan every element without re-deriving strides.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A masked 1-D ([`FlatLayout`]) view: a flat row-major scan over a [`Tile`].
pub type FlatView<'a, T> = MaskedView<'a, T, Coords1d>;
/// The mutable twin of [`FlatView`].
pub type FlatViewMut<'a, T> = MaskedViewMut<'a, T, Coords1d>;

/// Maps a flat row-major index to an N-D coordinate over `shape`: the inverse of a
/// strided dot. Re-view a [`Window`]ed [`View`](cubecl::std::tensor::View) through this to walk it
/// linearly (`shape()` is the element count) without re-deriving strides in the kernel.
#[derive(CubeType, Clone)]
pub struct FlatLayout {
    shape: CoordsDyn,
}

#[cube]
impl FlatLayout {
    pub fn new(shape: CoordsDyn) -> Self {
        FlatLayout { shape }
    }
}

#[cube]
impl Layout for FlatLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = self.shape.len().comptime();
        let mut out = CoordsDyn::new();
        let mut offs = pos as u32;

        // Peel off the least-significant dim each step (row-major), carrying the quotient up.
        #[unroll]
        for i in 0..rank {
            let dim = rank - i - 1;
            let extent = self.shape[dim];
            out.push(offs % extent);
            offs /= extent;
        }

        out.reverse(); // pushed last→first; restore ascending dim order
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        let mut total = 1u32;

        #[unroll]
        for p in 0..self.shape.len() {
            total *= self.shape[p];
        }

        total as usize
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.shape()
    }
}

#[cube]
impl<I: Numeric, N: Size> Tile<Vector<I, N>> {
    // the generic F type parameter is necessary for now as the `MemData` holds the same data type `T` as the `Tile` exposes
    // so we need to cast the value explicitly here. this should be fixed in the future when `MemData` is refactored.
    pub fn flat<F: Numeric>(&self) -> TileView<'_, I, N, Coords1d, F> {
        match &self.payload {
            Payload::Gmem(g) | Payload::Smem(g) => {
                let values = g.flat();
                let quant = self.quant;
                if comptime!(quant.is_some()) {
                    let info = quant.unwrap();
                    TileView::new_Quantized(QuantizedView::new(
                        values,
                        F::cast_from(info.scale),
                        comptime!(info.scheme),
                    ))
                } else {
                    TileView::new_Direct(values)
                }
            }
            Payload::Cmma(_) => panic!("Tile::flat: a cmma fragment has no memory view"),
        }
    }

    pub fn flat_mut<F: Numeric>(&mut self) -> TileViewMut<'_, I, N, Coords1d, F> {
        match &mut self.payload {
            Payload::Gmem(g) | Payload::Smem(g) => {
                let values = g.flat_mut();
                let quant = self.quant;
                if comptime!(quant.is_some()) {
                    panic!("writing to quantized view is not supported yet")
                } else {
                    TileViewMut::new_Direct(values)
                }
            }
            Payload::Cmma(_) => panic!("Tile::flat_mut: a cmma fragment has no memory view"),
        }
    }
}
