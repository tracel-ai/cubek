use cubecl::features::TypeUsage;
use cubecl::prelude::*;

use super::extrema::numeric_is_nan;
use crate::components::instructions::{ReduceOutputMode, Value};
use crate::components::precision::ReducePrecision;

/// A candidate's value and coordinate folded into one unsigned integer whose
/// unsigned order is the pair's order: the value in the key's [`ValueOrder`],
/// and the lower coordinate where two values are equal.
pub(crate) type OrderKey = u64;

const SIGN: u32 = 0x8000_0000;

/// Which end of the value range an [`OrderKey`] ranks first.
///
/// A NaN outranks every number in both, so neither is the other's reverse and a
/// key built for one cannot be read as the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ValueOrder {
    /// Largest value first, as top-k and max rank.
    Descending,
    /// Smallest value first, as min ranks.
    Ascending,
}

/// Whether an accumulation type packs into an [`OrderKey`] on this device: it
/// must leave room for a `u32` coordinate beside it, and the device must be
/// able to compare the resulting 64-bit integer (WGSL cannot).
#[cube]
pub(crate) fn packs_into_key<N: Numeric>() -> comptime_type!(bool) {
    let elem = elem_type_of::<N>();
    let properties = comptime::device_properties().comptime();

    comptime!(
        matches!(
            elem,
            ElemType::Float(_) | ElemType::Int(_) | ElemType::UInt(_)
        ) && elem.size_bits() == 32
            && properties
                .type_usage(u64::elem_type_native())
                .contains(TypeUsage::Arithmetic)
    )
}

/// Whether a coordinate-tracking reduction packs into an [`OrderKey`] on this device.
#[cube]
pub(crate) fn packs_key<P: ReducePrecision>(
    #[comptime] output: ReduceOutputMode,
) -> comptime_type!(bool) {
    let tracks_coordinates = comptime!(output.has_indices());
    let packs = packs_into_key::<P::EA>();

    comptime!(tracks_coordinates && packs)
}

#[cube]
pub(crate) fn pack_order_key<N: Numeric, S: Size>(
    value: Vector<N, S>,
    coordinate: Vector<u32, S>,
    #[comptime] order: ValueOrder,
) -> Vector<OrderKey, S> {
    // Inverted, so that a lower coordinate makes a larger key and wins a tie.
    let rank = Vector::new(u32::MAX) - coordinate;

    (Vector::<OrderKey, S>::cast_from(order_bits::<N, S>(value, order)) << Vector::new(32u64))
        | Vector::<OrderKey, S>::cast_from(rank)
}

/// The key of a slot that has taken no candidate: `last`, the value the unpacked
/// accumulator starts from, at coordinate `u32::MAX`, so a row with nothing to
/// rank reports the same value and index either way.
///
/// That coordinate's rank is zero, and it is written as zero rather than
/// computed: the subtraction of a constant from itself is folded by the WGSL
/// optimizer into a constant it cannot build for a vector type.
#[cube]
pub(crate) fn empty_order_key<N: Numeric, S: Size>(
    last: Vector<N, S>,
    #[comptime] order: ValueOrder,
) -> Vector<OrderKey, S> {
    Vector::<OrderKey, S>::cast_from(order_bits::<N, S>(last, order)) << Vector::new(32u64)
}

/// The better-ranked of two keys, whichever [`ValueOrder`] built them.
#[cube]
pub(crate) fn better_order_key<S: Size>(
    current: Vector<OrderKey, S>,
    candidate: Vector<OrderKey, S>,
) -> Vector<OrderKey, S> {
    select_many(current.greater_than(&candidate), current, candidate)
}

/// Replace a single-slot accumulator's key with whichever of it and `candidate`
/// ranks better, whichever [`ValueOrder`] built them.
#[cube]
pub(crate) fn key_insert<S: Size>(
    keys: &mut Value<Vector<OrderKey, S>>,
    candidate: Vector<OrderKey, S>,
) {
    let winning = better_order_key::<S>(keys.item(), candidate);
    keys.assign(&Value::new_single(winning));
}

/// Insert a candidate into `keys`, held in ranked order, dropping the last.
///
/// The ordering and the tie-break ride one comparison, and one selected pair
/// swaps both halves of a slot. Only a caller that is itself unrolled `k` times
/// asks for `unrolled`, since the budget it is priced against is the `k * k` of
/// that nest rather than this loop's `k`.
#[cube]
pub(crate) fn ranked_key_insert<S: Size>(
    keys: &mut Array<Vector<OrderKey, S>>,
    insert_key: Vector<OrderKey, S>,
    #[comptime] k: usize,
    #[comptime] unrolled: bool,
) {
    let mut insert_key = insert_key;

    #[unroll(unrolled)]
    for j in 0..k {
        let to_keep = keys[j].greater_than(&insert_key);
        let next_key = select_many(to_keep, insert_key, keys[j]);
        keys[j] = select_many(to_keep, keys[j], insert_key);
        insert_key = next_key;
    }
}

/// Collapse a vectorized accumulator's lanes down to the one winning key.
#[cube]
pub(crate) fn finalize_key<S: Size>(keys: Vector<OrderKey, S>) -> OrderKey {
    let vector_size = keys.vector_size().comptime();
    let mut winning = keys.extract(0usize);

    #[unroll]
    for k in 1..vector_size {
        let candidate = keys.extract(k);
        winning = select(winning > candidate, winning, candidate);
    }

    winning
}

#[cube]
pub(crate) fn order_key_value<N: Numeric, S: Size>(
    key: Vector<OrderKey, S>,
    #[comptime] order: ValueOrder,
) -> Vector<N, S> {
    value_from_order_bits::<N, S>(Vector::cast_from(key >> Vector::new(32u64)), order)
}

#[cube]
pub(crate) fn order_key_coordinate<S: Size>(key: Vector<OrderKey, S>) -> Vector<u32, S> {
    Vector::new(u32::MAX) - Vector::cast_from(key & Vector::new(0xFFFF_FFFFu64))
}

/// The value's bits mapped so that unsigned comparison of the results ranks the
/// values in `order`.
///
/// A float's sign bit orders backwards and its magnitude bits invert under it,
/// hence the flip. `-0.0` is mapped onto `+0.0`, since the two compare equal and
/// a key that told them apart would break the tie towards the wrong coordinate.
/// Every NaN takes the top key whatever its payload or sign, so NaNs outrank
/// every number and tie among themselves, leaving the coordinate to decide as
/// the instructions' policy states. A winning NaN therefore reads back as a
/// canonical NaN, not as its input bits.
#[cube]
fn order_bits<N: Numeric, S: Size>(
    value: Vector<N, S>,
    #[comptime] order: ValueOrder,
) -> Vector<u32, S> {
    let bits = Vector::<u32, S>::reinterpret(value);
    let sign = Vector::new(SIGN);
    let elem = elem_type_of::<N>();

    match comptime!(elem) {
        ElemType::Float(_) => {
            let zero = Vector::new(N::from_int(0));

            let ordered = match comptime!(order) {
                ValueOrder::Descending => select_many(
                    value.less_than(&zero),
                    Vector::new(u32::MAX) - bits,
                    bits | sign,
                ),
                ValueOrder::Ascending => {
                    select_many(value.greater_than(&zero), sign - bits, bits | sign)
                }
            };

            select_many(numeric_is_nan(value), Vector::new(u32::MAX), ordered)
        }
        ElemType::Int(_) => reversed_if_ascending::<S>(bits ^ sign, order),
        ElemType::UInt(_) => reversed_if_ascending::<S>(bits, order),
        _ => panic!("an order key packs floats, signed and unsigned integers only"),
    }
}

#[cube]
fn value_from_order_bits<N: Numeric, S: Size>(
    bits: Vector<u32, S>,
    #[comptime] order: ValueOrder,
) -> Vector<N, S> {
    let sign = Vector::new(SIGN);
    let elem = elem_type_of::<N>();

    let value_bits = match comptime!(elem) {
        ElemType::Float(_) => match comptime!(order) {
            ValueOrder::Descending => select_many(
                (bits & sign).equal(&Vector::new(0u32)),
                Vector::new(u32::MAX) - bits,
                bits ^ sign,
            ),
            // `<=` rather than a sign test, so that the one key both zeros share
            // reads back as `+0.0` here as it does in the descending arm.
            ValueOrder::Ascending => select_many(bits.less_equal(&sign), sign - bits, bits),
        },
        ElemType::Int(_) => reversed_if_ascending::<S>(bits, order) ^ sign,
        ElemType::UInt(_) => reversed_if_ascending::<S>(bits, order),
        _ => panic!("an order key packs floats, signed and unsigned integers only"),
    };

    Vector::<N, S>::reinterpret(value_bits)
}

/// Reverse an unsigned image that already rises with the value, so that it falls
/// with it instead. Its own inverse, so both directions of the map use it.
#[cube]
fn reversed_if_ascending<S: Size>(
    rising: Vector<u32, S>,
    #[comptime] order: ValueOrder,
) -> Vector<u32, S> {
    match comptime!(order) {
        ValueOrder::Descending => rising,
        ValueOrder::Ascending => Vector::new(u32::MAX) - rising,
    }
}
