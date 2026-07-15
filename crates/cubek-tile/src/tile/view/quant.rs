use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// The `Quantized` arm of a [`TileView`]: the storage element `I` the buffer truly holds, and the
/// scales as a second view over that same window (see [`ScaleLayout`]). Both are flat and take the
/// same [`Coords1d`], so a read pairs a line with its own scale by construction. Nothing here is
/// typed in the served element — that is only what a [`read`](QuantizedView::read) is asked for.
#[derive(CubeType)]
pub struct QuantizedView<'a, I: Numeric, W: Size> {
    values: FlatView<'a, Vector<I, W>>,
    scales: FlatView<'a, f32>,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, I: Numeric, W: Size> QuantizedView<'a, I, W> {
    pub fn new(
        values: FlatView<'a, Vector<I, W>>,
        scales: FlatView<'a, f32>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, I, W> {
            values,
            scales,
            scheme,
        }
    }

    pub fn read<O: Numeric>(&self, pos: Coords1d) -> Vector<O, W> {
        let raw = Vector::<O, W>::cast_from(self.values.read(pos));
        match comptime!(self.scheme.store) {
            QuantStore::Native => raw * Vector::cast_from(self.scales.read(pos)),
            _ => panic!("only native quantization storage is supported for now"),
        }
    }

    pub fn shape(&self) -> Coords1d {
        self.values.shape()
    }
}

/// The scales' [`GmemLayout`]: a window coordinate to the flat index of its block's scale, the dot
/// of each axis's block index with its scale stride. `window_start` carries the window origin's own
/// block (folded in at descent by [`QuantInfo`]), so this only adds the offset within the window —
/// sound because no window straddles a block, which
/// [`quantized`](crate::StridedTileArgLaunch::quantized) rejects at launch.
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
    /// Physical line width, so the inner axis's line coordinate scales back to elements.
    #[cube(comptime)]
    vector_size: usize,
}

#[cube]
impl ScaleLayout {
    pub fn new(
        strides: Coords<u32>,
        window_start: u32,
        #[comptime] block: Vec<usize>,
        #[comptime] vector_size: usize,
    ) -> Self {
        ScaleLayout {
            strides,
            window_start,
            block,
            vector_size,
        }
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
            // Only the innermost axis counts lines; blocks are cut in elements, so widen it.
            let w = comptime!((if p == last { self.vector_size } else { 1 }) as u32);
            let block = comptime!(self.block[p] as u32);
            terms.push(pos[p].fmul(w).fdiv(block).fmul(self.strides.at(p)));
        }
        self.window_start
            .fadd(terms.fsum(comptime!((0..rank).collect::<Vec<_>>())))
            .fcast::<usize>()
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
