//! Constness-preserving kernel arithmetic. A cubecl expand element already knows
//! whether it holds a constant (`Variable::Constant`), but the stock operators always
//! emit an instruction, so a computed constant degrades to a runtime value, the crack
//! every comptime twin field grew out of. These fold instead: constant operands compute
//! at expand time, identities pass through, so comptime-ness rides plain `u32`/`usize`
//! values through walks and layouts, and one code path serves both.

use cubecl::ir::{ConstantValue, Scope, Value};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::unexpanded;

/// Folding arithmetic on integer kernel values; `f` for folding.
pub trait Fold: Sized {
    /// `self + rhs`; `x + 0` passes through.
    fn fadd(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self - rhs`; `x - 0` passes through.
    fn fsub(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self * rhs`; `x * 1` passes through, `x * 0` is `0`.
    fn fmul(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self / rhs`; `x / 1` passes through.
    fn fdiv(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self % rhs`; `x % 1` is `0`.
    fn frem(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The value re-typed to `To`, a constant staying constant (the stock `as` emits a
    /// cast instruction, which erases constness).
    fn fcast<To: Int>(self) -> To {
        unexpanded!()
    }
    /// The comptime constant this value holds, if any: the bridge from a folded value
    /// back to host data (fragment selection needs host indices).
    fn constant(self) -> Option<u64> {
        unexpanded!()
    }
}

impl Fold for u32 {}
impl Fold for usize {}

/// Folding reductions over the elements at comptime `picks`: a sequence accumulates
/// by chaining fresh values, where a `let mut` accumulator would land in a mutable
/// slot and erase constness.
pub trait FoldSeq<C: Int>: Sized {
    /// Product of the picked elements (empty picks fold to `1`).
    fn fproduct(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }
    /// Sum of the picked elements (empty picks fold to `0`).
    fn fsum(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }
}

impl<C: Int + Fold> FoldSeq<C> for Sequence<C> {}

/// The constant a non-negative integer expand element holds, if any.
fn constant<C: Int>(e: &NativeExpand<C>) -> Option<u64> {
    match e.expand.as_const() {
        Some(ConstantValue::UInt(v)) => Some(v),
        Some(ConstantValue::Int(v)) if v >= 0 => Some(v as u64),
        _ => None,
    }
}

/// A constant expand element of `e`'s type holding `v`.
fn constant_like<C: Int>(v: u64, e: &NativeExpand<C>) -> NativeExpand<C> {
    Value::constant(v.into(), e.expand.value_type()).into()
}

fn fold_add<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(a + b, &lhs),
        (Some(0), None) => rhs,
        (None, Some(0)) => lhs,
        _ => AddExpand::__expand_add_method(lhs, scope, rhs),
    }
}

fn fold_sub<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if a >= b => constant_like(a - b, &lhs),
        (None, Some(0)) => lhs,
        _ => SubExpand::__expand_sub_method(lhs, scope, rhs),
    }
}

fn fold_mul<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(a * b, &lhs),
        (Some(0), None) | (None, Some(0)) => constant_like(0, &lhs),
        (Some(1), None) => rhs,
        (None, Some(1)) => lhs,
        _ => MulExpand::__expand_mul_method(lhs, scope, rhs),
    }
}

fn fold_div<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if b != 0 => constant_like(a / b, &lhs),
        (None, Some(1)) => lhs,
        // 0 / x is 0 for any in-range divisor (a divisor here is an extent, never 0).
        (Some(0), None) => constant_like(0, &lhs),
        _ => DivExpand::__expand_div_method(lhs, scope, rhs),
    }
}

fn fold_rem<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if b != 0 => constant_like(a % b, &lhs),
        (None, Some(1)) | (Some(0), None) => constant_like(0, &lhs),
        _ => RemExpand::__expand_rem_method(lhs, scope, rhs),
    }
}

/// Expand twin of [`Fold`]; blanket on integer expand elements.
pub trait FoldExpand<C: Int>: Sized {
    fn __expand_fadd_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_fsub_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_fmul_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_fdiv_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_frem_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_fcast_method<To: Int>(self, scope: &Scope) -> NativeExpand<To>;
    fn __expand_constant_method(self, scope: &Scope) -> Option<u64>;
}

impl<C: Int> FoldExpand<C> for NativeExpand<C> {
    fn __expand_fadd_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_add(scope, self, rhs)
    }
    fn __expand_fsub_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_sub(scope, self, rhs)
    }
    fn __expand_fmul_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_mul(scope, self, rhs)
    }
    fn __expand_fdiv_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_div(scope, self, rhs)
    }
    fn __expand_frem_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_rem(scope, self, rhs)
    }
    fn __expand_fcast_method<To: Int>(self, scope: &Scope) -> NativeExpand<To> {
        match constant(&self) {
            Some(v) => Value::constant(v.into(), To::__expand_as_type(scope)).into(),
            None => To::__expand_cast_from(scope, self),
        }
    }
    fn __expand_constant_method(self, _scope: &Scope) -> Option<u64> {
        constant(&self)
    }
}

/// Expand twin of [`FoldSeq`]; blanket on integer sequences.
pub trait FoldSeqExpand<C: Int>: Sized {
    fn __expand_fproduct_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C>;
    fn __expand_fsum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C>;
}

impl<C: Int> FoldSeqExpand<C> for SequenceExpand<C> {
    fn __expand_fproduct_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            Value::constant(1u64.into(), C::__expand_as_type(scope)).into();
        for i in picks {
            let e = *self.__expand_index_method(scope, NativeExpand::from_lit(scope, i));
            acc = fold_mul(scope, acc, e);
        }
        acc
    }

    fn __expand_fsum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            Value::constant(0u64.into(), C::__expand_as_type(scope)).into();
        for i in picks {
            let e = *self.__expand_index_method(scope, NativeExpand::from_lit(scope, i));
            acc = fold_add(scope, acc, e);
        }
        acc
    }
}

/// An immutable coordinate/extent list: [`CoordsDyn`]'s stored-data sibling, whose
/// expand's `IntoMut` is the identity. Elements are never reassigned after
/// construction, so a `let mut` holder (staging slot, windowed tile) must not copy
/// them into mutable slots; `Sequence` does, and that copy erases constness.
pub struct Coords<C: Int> {
    _c: core::marker::PhantomData<C>,
}

impl<C: Int> Clone for Coords<C> {
    fn clone(&self) -> Self {
        Coords {
            _c: core::marker::PhantomData,
        }
    }
}

#[allow(clippy::new_without_default, clippy::len_without_is_empty)]
impl<C: Int> Coords<C> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> Self {
        unexpanded!()
    }
    pub fn push(&mut self, _v: C) {
        unexpanded!()
    }
    /// The element at comptime `i`.
    pub fn at(&self, _i: usize) -> C {
        unexpanded!()
    }
    /// The comptime length.
    pub fn len(&self) -> usize {
        unexpanded!()
    }
    /// Re-view as boundary [`CoordsDyn`] (same handles; cubecl layouts flow those).
    pub fn to_dyn(&self) -> CoordsDyn {
        unexpanded!()
    }
    /// Product of the elements at comptime `picks` (empty picks fold to `1`).
    pub fn fproduct(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }
    /// Sum of the elements at comptime `picks` (empty picks fold to `0`).
    pub fn fsum(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }

    pub fn __expand_new(_scope: &Scope) -> CoordsExpand<C> {
        CoordsExpand { values: Vec::new() }
    }
}

pub struct CoordsExpand<C: Int> {
    values: Vec<NativeExpand<C>>,
}

impl<C: Int> CubeType for Coords<C> {
    type ExpandType = CoordsExpand<C>;
}

impl<C: Int> IntoExpand for CoordsExpand<C> {
    type Expand = Self;
    fn into_expand(self, _scope: &Scope) -> Self {
        self
    }
}

/// Identity: the whole point of the type (see [`Coords`]).
impl<C: Int> IntoMut for CoordsExpand<C> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<C: Int> CubeDebug for CoordsExpand<C> {}

impl<C: Int> Clone for CoordsExpand<C> {
    fn clone(&self) -> Self {
        CoordsExpand {
            values: self.values.clone(),
        }
    }
}

impl<C: Int> ExpandTypeClone for CoordsExpand<C> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl<C: Int> AsRefExpand for CoordsExpand<C> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}

impl<C: Int> AsMutExpand for CoordsExpand<C> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<C: Int> CoordsExpand<C> {
    pub fn __expand_push_method(&mut self, _scope: &Scope, v: NativeExpand<C>) {
        self.values.push(v);
    }
    pub fn __expand_at_method(&self, _scope: &Scope, i: NativeExpand<usize>) -> NativeExpand<C> {
        let i = i
            .expand
            .as_const()
            .expect("Coords::at: comptime index only")
            .as_i64() as usize;
        self.values[i]
    }
    pub fn __expand_len_method(&self, _scope: &Scope) -> usize {
        self.values.len()
    }
    pub fn __expand_to_dyn_method(&self, scope: &Scope) -> SequenceExpand<u32> {
        let mut out = Sequence::<u32>::__expand_new(scope);
        for v in &self.values {
            // Same handles, re-typed to the boundary element (u32 coordinates).
            out.__expand_push_method(scope, unsafe { *v.as_type_ref_unchecked::<u32>() });
        }
        out
    }
    pub fn __expand_fproduct_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            Value::constant(1u64.into(), C::__expand_as_type(scope)).into();
        for i in picks {
            acc = fold_mul(scope, acc, self.values[i]);
        }
        acc
    }
    pub fn __expand_fsum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            Value::constant(0u64.into(), C::__expand_as_type(scope)).into();
        for i in picks {
            acc = fold_add(scope, acc, self.values[i]);
        }
        acc
    }
}
