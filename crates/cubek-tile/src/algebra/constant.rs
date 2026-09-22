//! Integer arithmetic on kernel values that keeps a constant constant. An expand element knows
//! whether it is a constant (`Variable::Constant`), but the stock operators always emit an
//! instruction, degrading a computed constant to runtime. Here two constants compute at expand
//! time and identities pass through, so an index built from stated extents stays comptime.

use cubecl::ir::{
    ConstantValue, ExpandValue, Scope,
    interfaces::{ScalarType, TypedExt},
    try_cast_ty,
};
use cubecl::prelude::*;
use cubecl::unexpanded;

/// Arithmetic on an integer kernel value that a constant survives.
pub trait Known: Sized {
    /// `self + rhs`; `x + 0` passes through.
    fn plus(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self - rhs`; `x - 0` passes through.
    fn minus(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self * rhs`; `x * 1` passes through, `x * 0` is `0`.
    fn times(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self / rhs`; `x / 1` passes through.
    fn divided_by(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self % rhs`; `x % 1` is `0`.
    fn remainder(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The smaller of the two; two constants fold.
    fn min_with(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The larger of the two; two constants fold.
    fn max_with(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The value re-typed to `To`, a constant staying constant (the stock `as` emits a
    /// cast instruction, which erases constness).
    fn retyped<To: Int>(self) -> To {
        unexpanded!()
    }
    /// The comptime constant this value holds, if any: the bridge from a folded value
    /// back to host data (fragment selection needs host indices).
    fn constant(self) -> Option<u64> {
        unexpanded!()
    }
}

impl Known for u32 {}
impl Known for usize {}
impl Known for i32 {}

/// Constant-keeping sums over the elements at comptime `picks`: a sequence accumulates by
/// chaining fresh values, where a `let mut` accumulator would land in a mutable slot and erase
/// constness.
pub(crate) trait KnownSeq<C: Int>: Sized {
    /// Sum of the picked elements (empty picks fold to `0`).
    fn sum(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }
}

impl<C: Int + Known> KnownSeq<C> for Sequence<C> {}

/// The constant a non-negative integer expand element holds, if any.
pub(crate) fn constant<C: Int>(e: &NativeExpand<C>) -> Option<u64> {
    match e.expand.as_const() {
        Some(ConstantValue::UInt(v)) => Some(v),
        Some(ConstantValue::Int(v)) if v >= 0 => Some(v as u64),
        _ => None,
    }
}

/// A constant expand element of `e`'s type holding `v`.
fn constant_like<C: Int>(scope: &Scope, v: u64, e: &NativeExpand<C>) -> NativeExpand<C> {
    let ty = match e.expand {
        ExpandValue::Value(val) => {
            let ctx = scope.ctx();
            let scalar = val.try_get_scalar_elem_ty(ctx).unwrap().deref(ctx);
            try_cast_ty!(scalar, ctx, dyn ScalarType).elem_type(ctx)
        }
        ExpandValue::Constant { ty, .. } => ty,
    };
    ExpandValue::constant(v.into(), ty).into()
}

pub(crate) fn fold_add<C: Int>(
    scope: &Scope,
    lhs: NativeExpand<C>,
    rhs: NativeExpand<C>,
) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a + b, &lhs),
        (Some(0), None) => rhs,
        (None, Some(0)) => lhs,
        _ => AddExpand::__expand_add_method(lhs, scope, rhs),
    }
}

fn fold_sub<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if a >= b => constant_like(scope, a - b, &lhs),
        (None, Some(0)) => lhs,
        _ => SubExpand::__expand_sub_method(lhs, scope, rhs),
    }
}

pub(crate) fn fold_mul<C: Int>(
    scope: &Scope,
    lhs: NativeExpand<C>,
    rhs: NativeExpand<C>,
) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a * b, &lhs),
        (Some(0), None) | (None, Some(0)) => constant_like(scope, 0, &lhs),
        (Some(1), None) => rhs,
        (None, Some(1)) => lhs,
        _ => MulExpand::__expand_mul_method(lhs, scope, rhs),
    }
}

fn fold_div<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if b != 0 => constant_like(scope, a / b, &lhs),
        (None, Some(1)) => lhs,
        // 0 / x is 0 for any in-range divisor (a divisor here is an extent, never 0).
        (Some(0), None) => constant_like(scope, 0, &lhs),
        _ => DivExpand::__expand_div_method(lhs, scope, rhs),
    }
}

fn fold_rem<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if b != 0 => constant_like(scope, a % b, &lhs),
        (None, Some(1)) | (Some(0), None) => constant_like(scope, 0, &lhs),
        _ => RemExpand::__expand_rem_method(lhs, scope, rhs),
    }
}

fn fold_min<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a.min(b), &lhs),
        _ => C::__expand_min(scope, lhs, rhs),
    }
}

fn fold_max<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a.max(b), &lhs),
        _ => C::__expand_max(scope, lhs, rhs),
    }
}

/// Expand twin of [`Known`]; blanket on integer expand elements.
pub(crate) trait KnownExpand<C: Int>: Sized {
    fn __expand_plus_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_minus_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_times_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_divided_by_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_remainder_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_min_with_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_max_with_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_retyped_method<To: Int>(self, scope: &Scope) -> NativeExpand<To>;
    fn __expand_constant_method(self, scope: &Scope) -> Option<u64>;
}

impl<C: Int> KnownExpand<C> for NativeExpand<C> {
    fn __expand_plus_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_add(scope, self, rhs)
    }
    fn __expand_minus_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_sub(scope, self, rhs)
    }
    fn __expand_times_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_mul(scope, self, rhs)
    }
    fn __expand_divided_by_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_div(scope, self, rhs)
    }
    fn __expand_remainder_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_rem(scope, self, rhs)
    }
    fn __expand_min_with_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_min(scope, self, rhs)
    }
    fn __expand_max_with_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_max(scope, self, rhs)
    }
    fn __expand_retyped_method<To: Int>(self, scope: &Scope) -> NativeExpand<To> {
        match constant(&self) {
            Some(v) => ExpandValue::constant(v.into(), To::elem_type(scope)).into(),
            None => To::__expand_cast_from(scope, self),
        }
    }
    fn __expand_constant_method(self, _scope: &Scope) -> Option<u64> {
        constant(&self)
    }
}

/// Expand twin of [`KnownSeq`]; blanket on integer sequences.
pub(crate) trait KnownSeqExpand<C: Int>: Sized {
    fn __expand_sum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C>;
}

impl<C: Int> KnownSeqExpand<C> for SequenceExpand<C> {
    fn __expand_sum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            ExpandValue::constant(0u64.into(), C::elem_type(scope)).into();
        for i in picks {
            let e = *self.__expand_index_method(scope, NativeExpand::from_lit(scope, i));
            acc = fold_add(scope, acc, e);
        }
        acc
    }
}
