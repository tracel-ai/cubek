use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
    // `Coordinates` is referenced only as a bound (fully qualified below): importing the trait
    // would pull its `u32: Coordinates::from_int` into scope and clash with `Numeric::from_int`
    // in `unpack_lane`.
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// The `Quantized` arm of a [`TileView`]: the storage element `I` the buffer truly holds, and the
/// scales as a second view over that same window (see [`ScaleLayout`]). Both take the same
/// coordinate `C`, so a read pairs a line with its own scale by construction. `C` is
/// [`Coords1d`] for a flat scan ([`flat_transparent`](crate::MemData::flat_transparent)) or
/// [`Coords2d`](cubecl::std::tensor::layout::Coords2d) for a matrix leaf
/// ([`matrix_transparent`](crate::MemData::matrix_transparent)); the dequant is the same either
/// way. Nothing here is typed in the served element, only what a
/// [`read`](QuantizedView::read) is asked for.
///
/// `WP` is the *physical* line: `Vector<I, WP>`, what the buffer is grouped into. The *served*
/// line is named per [`read`](QuantizedView::read) instead of on the struct, since it is only
/// meaningful at a read: a packed store yields `WP · num_quants` values per line, a
/// [`Native`](QuantStore::Native) one exactly `WP`. Either way one physical line is one served
/// line, so a position means the same thing to the values and the scales; only the width of the
/// result changes.
#[derive(CubeType)]
pub struct QuantizedView<'a, I: Numeric, WP: Size, C: cubecl::std::tensor::layout::Coordinates + 'a>
{
    /// `pub(crate)` so [`TileView::check`](crate::TileView) can read the overhang flag off the
    /// values without a separate accessor.
    pub(crate) values: MaskedView<'a, Vector<I, WP>, C>,
    scales: MaskedView<'a, f32, C>,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, I: Numeric, WP: Size, C: cubecl::std::tensor::layout::Coordinates + 'a>
    QuantizedView<'a, I, WP, C>
{
    pub fn new(
        values: MaskedView<'a, Vector<I, WP>, C>,
        scales: MaskedView<'a, f32, C>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, I, WP, C> {
            values,
            scales,
            scheme,
        }
    }

    /// Dequantize the line at `pos` into `W` served values, `W` the physical line times the
    /// scheme's packing factor (the launch narrowed the binding by exactly that).
    ///
    /// One scale per line, broadcast across it: a line never straddles a block
    /// ([`quantized`](crate::StridedTileArgLaunch::quantized) rejects that at launch), so the
    /// [`ScaleLayout`] resolves the whole line's block in one read.
    pub fn read<O: Numeric, W: Size>(&self, pos: C) -> Vector<O, W> {
        let scale = O::cast_from(self.scales.read(pos.clone()));
        match comptime!(self.scheme.store) {
            // One element, one value: the served line is the physical line, scaled.
            QuantStore::Native => {
                Vector::<O, W>::cast_from(self.values.read(pos)) * Vector::new(scale)
            }
            // Each element carries several values, laid down in storage order.
            QuantStore::PackedU32(_) => {
                let raw = self.values.read(pos);
                let mut out = Vector::<O, W>::empty();
                #[unroll]
                for lane in 0..out.vector_size() {
                    let q = unpack_lane(&raw, lane, comptime!(self.scheme));
                    out.insert(lane, O::cast_from(q) * scale);
                }
                out
            }
            other => panic!(
                "QuantizedView::read: quant storage {:?} is not wired",
                other
            ),
        }
    }

    pub fn shape(&self) -> C {
        self.values.shape()
    }
}

/// Served lane `lane` of a packed line: value `lane % pack` of word `lane / pack`,
/// sign-extended (anything `≥ 2^(bits-1)` folds to `value - 2^bits`, two's complement).
#[cube]
fn unpack_lane<I: Numeric, WP: Size>(
    line: &Vector<I, WP>,
    #[comptime] lane: usize,
    #[comptime] scheme: QuantScheme,
) -> i32 {
    let bits = comptime!(scheme.size_bits_value());
    let pack = comptime!(scheme.num_quants());
    let mask = u32::from_int((1 << bits) - 1);
    let sign_bit = u32::from_int(1 << (bits - 1));
    let two_pow_bits = 1 << bits;

    // The store is a `u32` by construction, whatever `I` the buffer was erased to; shifting
    // needs a concrete integer, so recover it here.
    let word = u32::cast_from(line.extract(comptime!(lane / pack)));
    let raw = (word >> u32::cast_from(comptime!(lane % pack * bits))) & mask;
    i32::cast_from(raw) - i32::cast_from(raw >= sign_bit) * two_pow_bits
}

/// The scales' [`GmemLayout`]: a window coordinate to the flat index of its block's scale, the dot
/// of each axis's block index with its scale stride. `window_start` carries the window origin's own
/// block (folded in at descent by [`QuantInfo`]), so this only adds the offset within the window,
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
    /// Served values per line, so the inner axis's line coordinate scales back to elements.
    /// Blocks are cut in values, so this is the *served* width; for a packed store it exceeds
    /// the physical line by the packing factor.
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
