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
///
/// `WP` is the *physical* line: `Vector<I, WP>`, what the buffer is grouped into. The *served*
/// line is named per [`read`](QuantizedView::read) instead of on the struct, since it is only
/// meaningful at a read: a packed store yields `WP · num_quants` values per line, a
/// [`Native`](QuantStore::Native) one exactly `WP`. Either way one physical line is one served
/// line, so a position means the same thing to the values, the scales and the [`FlatLayout`]
/// above them — only the width of the result changes.
#[derive(CubeType)]
pub struct QuantizedView<'a, I: Numeric, WP: Size> {
    values: FlatView<'a, Vector<I, WP>>,
    scales: FlatView<'a, f32>,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, I: Numeric, WP: Size> QuantizedView<'a, I, WP> {
    pub fn new(
        values: FlatView<'a, Vector<I, WP>>,
        scales: FlatView<'a, f32>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, I, WP> {
            values,
            scales,
            scheme,
        }
    }

    /// Dequantize the line at `pos` into `W` served values, `W` the physical line times the
    /// scheme's packing factor (asserted at expand).
    ///
    /// One scale per line, broadcast across it: a line never straddles a block
    /// ([`quantized`](crate::StridedTileArgLaunch::quantized) rejects that at launch), so the
    /// [`ScaleLayout`] resolves the whole line's block in one read.
    pub fn read<O: Numeric, W: Size>(&self, pos: Coords1d) -> Vector<O, W> {
        let scale = O::cast_from(self.scales.read(pos));
        match comptime!(self.scheme.store) {
            // One element, one value: the served line is the physical line, scaled.
            QuantStore::Native => {
                Vector::<O, W>::cast_from(self.values.read(pos)) * Vector::new(scale)
            }
            // Each element carries `num_quants` values: unpack every physical lane in turn and
            // lay its values down contiguously, so served lane `i · pack + j` is value `j` of
            // element `i` — the order the buffer already has.
            QuantStore::PackedU32(_) => {
                let raw = self.values.read(pos);
                let mut out = Vector::<O, W>::empty();
                let pack = comptime!(self.scheme.num_quants());
                // `.vector_size()` reads a width where `Size::value()` cannot go (it is
                // unexpanded in a `#[cube]` body — calling it kills the trace).
                let wp = raw.vector_size();
                comptime!(assert!(
                    out.vector_size() == wp * pack,
                    "QuantizedView::read: the served line must be the physical line ({wp}) \
                     times the scheme's packing factor ({pack})"
                ));
                let size!(P) = pack;
                #[unroll]
                for i in 0..wp {
                    // The store is a `u32` by construction, whatever `I` the buffer was erased
                    // to; shifting needs a concrete integer, so recover it here.
                    let values =
                        unpack_q::<O, P>(u32::cast_from(raw.extract(i)), comptime!(self.scheme));
                    #[unroll]
                    for j in 0..pack {
                        out.insert(comptime!(i * pack + j), values.extract(j) * scale);
                    }
                }
                out
            }
            other => panic!(
                "QuantizedView::read: quant storage {:?} is not wired",
                other
            ),
        }
    }

    pub fn shape(&self) -> Coords1d {
        self.values.shape()
    }
}

/// Unpack one packed `u32` into its `NF` quantized values, sign-extended.
///
/// Value `j` lives in bits `[j · size_quant, (j+1) · size_quant)`; anything `≥ 2^(bits-1)` folds
/// to `value - 2^bits` branchlessly (two's complement). Twin of `cubek-quant`'s `unpack_q` —
/// forked rather than shared, since that crate depends on this one.
#[cube]
pub(crate) fn unpack_q<O: Numeric, NF: Size>(
    value: u32,
    #[comptime] scheme: QuantScheme,
) -> Vector<O, NF> {
    let size_quant = comptime!(scheme.size_bits_value());
    let num_quant = comptime!(scheme.num_quants());

    let mask = u32::from_int((1 << size_quant) - 1);
    let sign_bit = u32::from_int(1 << (size_quant - 1));
    let two_pow_n = 1 << size_quant;

    let mut output = Vector::<O, NF>::empty();
    comptime!(assert!(
        output.vector_size() == num_quant,
        "unpack_q: the output line must hold the store's {num_quant} values"
    ));
    #[unroll]
    for position in 0..num_quant {
        let offset = u32::cast_from(comptime!(position * size_quant));
        let raw = (value >> offset) & mask;

        let raw_i32 = i32::cast_from(raw);
        let is_negative = i32::cast_from(raw >= sign_bit);
        output.insert(position, O::cast_from(raw_i32 - is_negative * two_pow_n));
    }
    output
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
    /// Served values per line, so the inner axis's line coordinate scales back to elements.
    /// Blocks are cut in values, so this is the *served* width — for a packed store it exceeds
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
