use crate::reduce::Accumulable;
use cubecl::prelude::*;

/// What a reduction accumulates its lines into, as its launch picks it, before the elements and
/// the width its kernel reads are known: one for each [`ReduceCarrier`](crate::reduce::ReduceCarrier).
pub trait ReduceCarrierFamily: Send + core::marker::Sync + 'static {
    /// The carrier of lines of `N` values of `Acc`, `K` values kept of each row.
    type Carrier<Acc: Numeric, N: Size, K: Size>: Accumulable<Acc, N> + Clone;
}
