use cubecl::{
    prelude::*,
    // `Coordinates` is referenced only as a bound (fully qualified below): importing the trait
    // would pull its `u32: Coordinates::from_int` into scope and clash with `Numeric::from_int`.
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// The scales' [`GmemLayout`]: a window coordinate to the flat index of its block's scale, the dot
/// of each axis's block index with its scale stride. `window_start` carries the window origin's own
/// block (folded in at descent by [`QuantInfo`]), so this only adds the offset within the window,
/// sound because no window straddles a block, which
/// [`quantized`](crate::StridedTileSource::quantized) rejects at launch.
///
/// Per-tensor never leaves index `0`: its strides are `0`, so every term folds away
/// ([`fmul`](crate::Fold::fmul) annihilates) and a read is a constant-index broadcast.
#[derive(CubeType, Clone)]
pub struct ScaleLayout {
    strides: Coords<u32>,
    window_start: u32,
    /// Per-axis block edges, in elements.
    #[cube(comptime)]
    block: Vec<usize>,
    /// Served values per line, so the inner axis's line coordinate scales back to elements.
    /// Blocks are cut in values, so this is the *served* width; for a packed store it exceeds
    /// the physical line by the packing factor.
    #[cube(comptime)]
    vector_size: usize,
    /// Per-axis window extent in elements, [`usize::MAX`] where it is not comptime.
    #[cube(comptime)]
    extent: Vec<usize>,
}

#[cube]
impl ScaleLayout {
    pub fn new(
        strides: Coords<u32>,
        window_start: u32,
        #[comptime] block: Vec<usize>,
        #[comptime] vector_size: usize,
        #[comptime] extent: Vec<usize>,
    ) -> Self {
        ScaleLayout {
            strides,
            window_start,
            block,
            vector_size,
            extent,
        }
    }

    /// Whether axis `p` still distinguishes scales: an extent that fits inside a block leaves one
    /// scale for the whole window, already folded into `window_start`, so the term is dropped at
    /// comptime rather than dividing a runtime coordinate that can only answer `0`.
    fn addresses(&self, #[comptime] p: usize) -> comptime_type!(bool) {
        comptime!(self.extent[p] > self.block[p])
    }
}

#[cube]
impl Layout for ScaleLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = comptime!(self.block.len());
        let last = comptime!(rank - 1);
        // Per-axis terms, summed below (chained, so a static window's dot folds).
        let mut terms = Sequence::<u32>::new();
        #[unroll]
        for p in 0..rank {
            if self.addresses(p) {
                // Only the innermost axis counts lines; blocks are cut in elements, so widen it.
                let w = comptime!((if p == last { self.vector_size } else { 1 }) as u32);
                let block = comptime!(self.block[p] as u32);
                terms.push(pos[p].fmul(w).fdiv(block).fmul(self.strides.at(p)));
            }
        }
        let kept = terms.len();
        if comptime!(kept == 0) {
            // Every axis holds one scale: the window's own, already in `window_start`.
            self.window_start.fcast::<usize>()
        } else {
            self.window_start
                .fadd(terms.fsum(comptime!((0..kept).collect::<Vec<_>>())))
                .fcast::<usize>()
        }
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        // The scales cover whatever window the values do, and the `FlatLayout` on top answers
        // for it; this layout only resolves an address.
        panic!("ScaleLayout::shape: the scales take the values' shape")
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // Nothing to clip: the values mask their own overhang, and the scales slice
        // bounds-checks the resolved index (a masked read there yields scale `0`).
        true.runtime()
    }
}
