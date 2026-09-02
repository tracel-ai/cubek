//! The scale-free unpacking read: a stored `u32`'s fields served as values.
//!
//! A packed operand is values and nothing else, so unpacking is not dequantization missing a
//! scale: it is the whole read. [`Packing::Packed`](crate::Packing::Packed) names the field, this view unpacks it, and no
//! scheme, scale binding or block grid is anywhere in the path. What folds a scale back in, where
//! there is one, is a verb the kernel writes ([`Tile::mm_scaled`](crate::Tile::mm_scaled)).

use std::marker::PhantomData;

use cubecl::ir::VectorSize;
use cubecl::prelude::barrier::Barrier;
use cubecl::quant::scheme::QuantValue;
use cubecl_common::e2m1x2;

use crate::{FieldDecode, field_decode};
use cubecl::unexpanded;
use cubecl::{
    prelude::*,
    std::tensor::{View, ViewExpand, ViewOperations, ViewOperationsExpand, layout::Coordinates},
};

/// Unpack one line of stored words into the `NF` values it holds: `NQ` words, each carrying
/// `32 / field.size_bits()` consecutive fields.
///
/// Two shapes, because a field decodes two ways. A `Q*` field is an integer: the width says how
/// many bits one holds and the top one is its sign, so `Q4S` is `[-8, 7]` in four bits. An `e2m1`
/// field is a float code, read back by reinterpreting the byte two of them share.
#[cube]
pub(crate) fn unpack_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: QuantValue,
) -> Vector<F, NF> {
    match comptime!(field_decode(field)) {
        FieldDecode::SignExtended => unpack_int_line::<F, NQ, NF>(words, field),
        FieldDecode::Reinterpreted => unpack_fp4_line::<F, NQ, NF>(words),
        // A launch states its field, so it can refuse this one before it compiles a kernel around
        // it; reaching here means nothing did.
        FieldDecode::Unserved => panic!(
            "unpack_line: {field:?} is not served by the packed view. Bind the tensor at its own \
             element and let the contraction cast it"
        ),
    }
}

/// The integer fields, sign-extended out of their slots.
#[cube]
fn unpack_int_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: QuantValue,
) -> Vector<F, NF> {
    let bits = comptime!(field.size_bits());
    let factor = comptime!(u32::BITS as usize / bits);
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

/// The `e2m1` fields, reinterpreted a pair at a time.
///
/// The pair is the unit rather than the value: two `e2m1` codes share a byte, and `e2m1x2` is what
/// the hardware decodes and what the shared software decoder names, so a byte answers two values
/// in one cast either way. That is [`QuantValue::native_packing`], read here as the loop's step.
#[cube]
fn unpack_fp4_line<F: Numeric, NQ: Size, NF: Size>(words: Vector<u32, NQ>) -> Vector<F, NF> {
    let pair = comptime!(QuantValue::E2M1.native_packing());
    let bytes = comptime!(u32::BITS as usize / (QuantValue::E2M1.size_bits() * pair));
    let factor = comptime!(bytes * pair);

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * factor;
        #[unroll]
        for j in 0..bytes {
            let byte = (word >> comptime!((j * 8) as u32)) & 0xff;
            let values = Vector::<F, Const<2>>::cast_from(e2m1x2::from_bits(byte as u8));
            out.insert(base + j * pair, values.extract(0usize));
            out.insert(base + j * pair + 1, values.extract(1usize));
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
    field: QuantValue,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> PackedView<'a, NQ, F, NF, C> {
    pub fn new(words: View<'a, Vector<u32, NQ>, C>, #[comptime] field: QuantValue) -> Self {
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
