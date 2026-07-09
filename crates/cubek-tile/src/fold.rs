//! Constness-preserving kernel arithmetic. A cubecl expand element already knows
//! whether it holds a constant (`Variable::Constant`), but the stock operators always
//! emit an instruction, so a computed constant degrades to a runtime value — the crack
//! every comptime twin field grew out of. These fold instead: constant operands compute
//! at expand time, identities pass through, so comptime-ness rides plain `u32`/`usize`
//! values through walks and layouts, and one code path serves both.

use cubecl::ir::{ConstantValue, Scope, Value};
use cubecl::prelude::*;
use cubecl::unexpanded;

/// Folding arithmetic on integer kernel values; `f` for folding.
pub trait Fold: Sized {
    /// `self + rhs`; `x + 0` passes through.
    fn fadd(self, _rhs: Self) -> Self {
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

/// Product of the elements at comptime `picks` (empty picks fold to `1`).
pub trait FoldSeq<C: Int>: Sized {
    fn fproduct(&self, _picks: Vec<usize>) -> C {
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
}
