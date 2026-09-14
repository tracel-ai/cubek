//! The scale-free unpacking read: a stored `u32`'s fields served as values.
//!
//! A packed operand is values and nothing else, so unpacking is not dequantization missing a
//! scale: it is the whole read. [`Packing::Packed`](crate::Packing::Packed) names the field, this view unpacks it, and no
//! scheme, scale binding or block grid is anywhere in the path. What folds a scale back in, where
//! there is one, is a verb the kernel writes ([`Tile::mm_scaled`](crate::Tile::mm_scaled)).
//!
//! [`SubwordView`] is the same read serving fewer values than a word holds: the word is read
//! whole and the slot picked at the read, for a reader stepping one value at a time.

use std::marker::PhantomData;

use cubecl::ir::FloatKind;
use cubecl::ir::VectorSize;
use cubecl::ir::types::Fp8Format;
use cubecl::post_processing::minifloat::{fp8_bits_to_f32, ue8m0_bits_to_f32};
use cubecl::prelude::barrier::Barrier;
use cubecl::quant::scheme::QuantValue;
use cubecl::std::quant::fp4::e2m1_packed_bits_to_float;
use cubecl::std::tensor::layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand};

use crate::{Field, FieldDecode, GmemLayout, Window, field_decode, float_field_bits};
use cubecl::unexpanded;
use cubecl::{
    prelude::*,
    std::tensor::{View, ViewExpand, ViewOperations, ViewOperationsExpand, layout::Coordinates},
};

/// Unpack one line of stored words into the `NF` values it holds: `NQ` words, each carrying
/// `NF / NQ` consecutive fields from its low bits, the whole word unless the line is a sub-word
/// read.
///
/// Four shapes, because a field decodes four ways. A `Q*` field is an integer: the width says
/// how many bits one holds and the top one is its sign, so `Q4S` is `[-8, 7]` in four bits. An
/// `e2m1` field is a float code, read back by reinterpreting the byte two of them share. An 8-bit
/// float code is a byte, read back through its format's decoder. A whole float is its own bits,
/// reinterpreted out of the slot it sits in.
#[cube]
pub(crate) fn unpack_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: Field,
) -> Vector<F, NF> {
    match comptime!(field_decode(field)) {
        FieldDecode::SignExtended => unpack_int_line::<F, NQ, NF>(words, field),
        FieldDecode::Reinterpreted => unpack_fp4_line::<F, NQ, NF>(words),
        FieldDecode::Byte(format) => unpack_byte_line::<F, NQ, NF>(words, format),
        FieldDecode::Bits(kind) => unpack_float_line::<F, NQ, NF>(words, kind),
    }
}

/// Fields one word contributes to an `nf`-wide line of `nq` words: the whole word, or the low
/// slots of a sub-word read.
fn fields_per_word(nq: usize, nf: usize, bits: usize) -> usize {
    let per_word = u32::BITS as usize / bits;
    let fields = nf / nq;
    assert!(
        fields > 0 && fields <= per_word && nf == fields * nq,
        "unpack_line: {nf} values from {nq} words of {per_word} fields"
    );
    fields
}

/// The integer fields, sign-extended out of their slots.
#[cube]
fn unpack_int_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: Field,
) -> Vector<F, NF> {
    let bits = comptime!(field.size_bits());
    let nq = NQ::value();
    let nf = NF::value();
    let factor = comptime!(fields_per_word(nq, nf, bits));
    let mask = comptime!(((1u64 << bits) - 1) as u32);
    // The sign bit, doubling as the bias of the branchless extension below.
    let sign = comptime!(1u32 << (bits - 1));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * factor;
        #[unroll]
        for j in 0..factor {
            let raw = (word >> comptime!((j * bits) as u32)) & mask;
            // Branchless sign extension: `(raw ^ s) - s` with `s = 2^(bits-1)` runs the identical
            // xor/sub on every lane, two uniform vector ops instead of a compare/select chain.
            let value = (raw ^ sign) as i32 - sign as i32;
            out.insert(base + j, F::cast_from(value));
        }
    }
    out
}

/// The `e2m1` fields, decoded a pair at a time.
///
/// The pair is the unit rather than the value: two `e2m1` codes share a byte, which is
/// [`QuantValue::native_packing`], read here as the loop's step.
///
/// Decoded in software, as `cubek-quant`'s field read and cubecl's own quantized view both are.
/// The `e2m1x2` cast lowers on CUDA alone, so a read reaching for it compiles on one vendor and
/// dies in codegen everywhere else — on a worker thread, which surfaces as a zeroed output
/// rather than as an error.
#[cube]
fn unpack_fp4_line<F: Numeric, NQ: Size, NF: Size>(words: Vector<u32, NQ>) -> Vector<F, NF> {
    let pair = comptime!(QuantValue::E2M1.native_packing());
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, QuantValue::E2M1.size_bits()));
    // A sub-word read of one code still decodes its pair and keeps the first.
    let bytes = comptime!(fields.div_ceil(pair));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * fields;
        #[unroll]
        for j in 0..bytes {
            let byte = (word >> comptime!((j * 8) as u32)) & 0xff;
            let values = e2m1_packed_bits_to_float::<F, Const<2>>(byte);
            out.insert(base + j * pair, values.extract(0usize));
            if comptime!(j * pair + 1 < fields) {
                out.insert(base + j * pair + 1, values.extract(1usize));
            }
        }
    }
    out
}

/// The 8-bit float codes, one byte each, through the format's decoder on `u32` patterns.
#[cube]
fn unpack_byte_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] format: Fp8Format,
) -> Vector<F, NF> {
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, 8));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * fields;
        #[unroll]
        for j in 0..fields {
            let byte = Vector::<u32, Const<1>>::new((word >> comptime!((j * 8) as u32)) & 0xff);
            let value = match comptime!(format) {
                Fp8Format::UE8M0 => ue8m0_bits_to_f32::<Const<1>>(byte),
                _ => fp8_bits_to_f32::<Const<1>>(byte, format),
            };
            out.insert(base + j, F::cast_from(value.extract(0usize)));
        }
    }
    out
}

/// One `f16` bit pattern, decoded to `f32` in integer arithmetic. Bits above the low half are
/// ignored.
///
/// Written out rather than reinterpreted through a 16-bit scalar, which not every target admits:
/// cubecl's own minifloat decoders are the same shape for the same reason.
#[cube]
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u32) << 16u32;
    let exponent = (bits >> 10u32) & 0x1fu32;
    let mantissa = bits & 0x3ffu32;
    // `f16` biases its exponent by 15 and `f32` by 127, and the mantissa moves up to `f32`'s 23
    // bits. A zero exponent is a subnormal, counted in steps of 2^-24.
    let normal = sign | ((exponent + 112u32) << 23u32) | (mantissa << 13u32);
    let subnormal = sign | u32::reinterpret(f32::cast_from(mantissa) * 5.9604645e-8f32);
    let finite = select(exponent == 0u32, subnormal, normal);
    let special = select(
        mantissa == 0u32,
        sign | 0x7f80_0000u32,
        sign | 0x7fc0_0000u32,
    );
    f32::reinterpret(select(exponent == 31u32, special, finite))
}

/// The whole floats, each read out of the slot it occupies: one `f32` a word, two `f16` or
/// `bf16`. A scale stored at its own precision is read here, which is what lets one word-typed
/// binding serve every scale a scheme can carry.
#[cube]
fn unpack_float_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] kind: FloatKind,
) -> Vector<F, NF> {
    let bits = comptime!(float_field_bits(kind));
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, bits));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * fields;
        #[unroll]
        for j in 0..fields {
            let slot = word >> comptime!((j * bits) as u32);
            // `bf16` is an `f32` truncated to its top half, so putting it back is the whole
            // decode. Every kind but these three was refused by `float_field_bits`.
            let value = match comptime!(kind) {
                FloatKind::F32 => F::cast_from(f32::reinterpret(slot)),
                FloatKind::F16 => F::cast_from(f16_bits_to_f32(slot)),
                _ => F::cast_from(f32::reinterpret((slot & 0xffffu32) << 16u32)),
            };
            out.insert(base + j, value);
        }
    }
    out
}

/// A [`View`] over stored words that serves the values they hold: reads `Vector<u32, NQ>` lines
/// and answers `Vector<F, NF>` ones, `NF = NQ * factor`.
///
/// The unscaled twin of cubecl's `QuantizedView`, and the reason it is a separate type rather than
/// that one with a scale of `1`: a view that takes a scale takes a scheme, a scale binding and a
/// block grid with it, and a packed operand has none of those to give.
#[expect(dead_code, reason = "read through the expand impls below")]
#[derive(CubeType, Clone)]
pub(crate) struct PackedView<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> {
    words: View<'a, Vector<u32, NQ>, C>,
    #[cube(comptime)]
    field: Field,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> PackedView<'a, NQ, F, NF, C> {
    pub fn new(words: View<'a, Vector<u32, NQ>, C>, #[comptime] field: Field) -> Self {
        PackedView::<'a, NQ, F, NF, C> {
            words,
            field,
            _ty: PhantomData,
        }
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> PackedView<'a, NQ, F, NF, C> {
    /// This view as a plain [`View`] of served values, which is what every reader takes.
    pub fn view(self) -> View<'a, Vector<F, NF>, C> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &Scope,
        this: PackedViewExpand<'a, NQ, F, NF, C>,
    ) -> ViewExpand<'a, Vector<F, NF>, C> {
        this.__expand_view_method(scope)
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static>
    PackedViewExpand<'a, NQ, F, NF, C>
{
    fn unpack(
        &self,
        scope: &Scope,
        words: NativeExpand<Vector<u32, NQ>>,
    ) -> NativeExpand<Vector<F, NF>> {
        unpack_line::expand::<F, NQ, NF>(scope, words, self.field)
    }

    pub fn __expand_view_method(self, scope: &Scope) -> ViewExpand<'a, Vector<F, NF>, C> {
        ViewExpand::new(scope, self)
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> Vectorized
    for PackedView<'a, NQ, F, NF, C>
{
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> VectorizedExpand
    for PackedViewExpand<'a, NQ, F, NF, C>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.words.__expand_vector_size_method(scope)
            * (u32::BITS as usize / self.field.size_bits())
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> ViewOperations<Vector<F, NF>, C>
    for PackedView<'a, NQ, F, NF, C>
{
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperationsExpand<Vector<F, NF>, C> for PackedViewExpand<'a, NQ, F, NF, C>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        let words = self.words.clone().__expand_read_method(scope, pos);
        self.unpack(scope, words)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        let words = self.words.clone().__expand_read_checked_method(scope, pos);
        self.unpack(scope, words)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        mask_value: NativeExpand<Vector<F, NF>>,
    ) -> NativeExpand<Vector<F, NF>> {
        let words = self
            .words
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);
        let value = self.unpack(scope, words);
        select::expand::<Vector<F, NF>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        let words = self
            .words
            .clone()
            .__expand_read_unchecked_method(scope, pos);
        self.unpack(scope, words)
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: <C>::ExpandType,
        _end: <C>::ExpandType,
    ) -> &SliceExpand<Vector<F, NF>> {
        panic!("PackedView: a packed operand has no raw slice of served values")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <C>::ExpandType {
        self.words.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<bool> {
        self.words.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<F, NF>>,
        _pos: C::ExpandType,
    ) {
        panic!("PackedView: a tensor map cannot unpack on the fly")
    }
}

/// The storage layout under a sub-word operand: a line index in served lines maps to the word
/// holding it, so the base and window layouts above it keep addressing lines and the buffer is
/// still read a word at a time.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct WordOfLine {
    lines: usize,
    #[cube(comptime)]
    per_line: usize,
}

#[cube]
impl WordOfLine {
    pub fn new(words: usize, #[comptime] per_line: usize) -> Self {
        WordOfLine {
            lines: words * per_line,
            per_line,
        }
    }
}

#[cube]
impl Layout for WordOfLine {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        pos / self.per_line
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.lines
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.lines
    }
}

/// One word read whole, `NF` of its fields served: the slot is the line's position in its word,
/// which is the line's index through the whole layout chain, so a row of scales need not start
/// on a word boundary. A reader stepping one value at a time under a runtime index reads the
/// word it lands in and shifts.
#[cube]
pub(crate) fn read_subword<
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn>,
>(
    words: &View<'_, Vector<u32, Const<1>>, Coords2d>,
    layout: &L,
    window: &Window,
    base: &GmemLayout,
    pos: Coords2d,
    #[comptime] field: Field,
    #[comptime] checked: bool,
) -> Vector<F, NF> {
    let nf = NF::value();
    let width = comptime!(nf as u32);
    let per_line = comptime!(field.per_word() / nf);
    let bits = comptime!(field.size_bits() as u32);
    let word = if checked {
        words.read_checked(pos)
    } else {
        words.read_unchecked(pos)
    };
    // A field that fills the word it is read from has no slot to find, and an `f32` scale is
    // exactly that: it costs the address walk below nothing.
    let shifted = if comptime!(per_line > 1) {
        let line = base.to_source_pos(window.to_source_pos(layout.to_source_pos(pos)));
        let slot = (line % per_line) as u32;
        word >> Vector::<u32, Const<1>>::new(slot * width * bits)
    } else {
        word
    };
    unpack_line::<F, Const<1>, NF>(shifted, field)
}

/// [`PackedView`] serving fewer values than a word holds ([`Packing::Subword`](crate::Packing::Subword)):
/// a `(row, col)` view in served lines whose words are addressed through [`WordOfLine`], and
/// whose slot is read off the same layout chain.
#[expect(dead_code, reason = "read through the expand impls below")]
#[derive(CubeType, Clone)]
pub(crate) struct SubwordView<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone,
> {
    words: View<'a, Vector<u32, Const<1>>, Coords2d>,
    layout: L,
    window: Window,
    base: GmemLayout,
    #[cube(comptime)]
    field: Field,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> SubwordView<'a, F, NF, L>
{
    pub fn new(
        words: View<'a, Vector<u32, Const<1>>, Coords2d>,
        layout: L,
        window: Window,
        base: GmemLayout,
        #[comptime] field: Field,
    ) -> Self {
        SubwordView::<'a, F, NF, L> {
            words,
            layout,
            window,
            base,
            field,
            _ty: PhantomData,
        }
    }
}

impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> SubwordView<'a, F, NF, L>
{
    pub fn view(self) -> View<'a, Vector<F, NF>, Coords2d> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &Scope,
        this: SubwordViewExpand<'a, F, NF, L>,
    ) -> ViewExpand<'a, Vector<F, NF>, Coords2d> {
        this.__expand_view_method(scope)
    }
}

impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> SubwordViewExpand<'a, F, NF, L>
{
    pub fn __expand_view_method(self, scope: &Scope) -> ViewExpand<'a, Vector<F, NF>, Coords2d> {
        ViewExpand::new(scope, self)
    }

    fn read(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
        checked: bool,
    ) -> NativeExpand<Vector<F, NF>> {
        read_subword::expand::<F, NF, L>(
            scope,
            &self.words,
            &self.layout,
            &self.window,
            &self.base,
            pos,
            self.field,
            checked,
        )
    }
}

impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> Vectorized for SubwordView<'a, F, NF, L>
{
}

impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> VectorizedExpand for SubwordViewExpand<'a, F, NF, L>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        VectorSize::from(NF::__expand_value(scope))
    }
}

impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> ViewOperations<Vector<F, NF>, Coords2d> for SubwordView<'a, F, NF, L>
{
}

impl<
    'a,
    F: Numeric,
    NF: Size,
    L: Layout<Coordinates = Coords2d, SourceCoordinates = CoordsDyn> + Clone + 'a,
> ViewOperationsExpand<Vector<F, NF>, Coords2d> for SubwordViewExpand<'a, F, NF, L>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        self.read(scope, pos, false)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        self.read(scope, pos, true)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
        mask_value: NativeExpand<Vector<F, NF>>,
    ) -> NativeExpand<Vector<F, NF>> {
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);
        let value = self.read(scope, pos, true);
        select::expand::<Vector<F, NF>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        self.read(scope, pos, false)
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: <Coords2d as CubeType>::ExpandType,
        _end: <Coords2d as CubeType>::ExpandType,
    ) -> &SliceExpand<Vector<F, NF>> {
        panic!("SubwordView: a packed operand has no raw slice of served values")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <Coords2d as CubeType>::ExpandType {
        self.words.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<bool> {
        self.words.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<F, NF>>,
        _pos: <Coords2d as CubeType>::ExpandType,
    ) {
        panic!("SubwordView: a tensor map cannot unpack on the fly")
    }
}
