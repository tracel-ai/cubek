use cubecl::features::TypeUsage;
use cubecl::prelude::*;

/// A candidate's value and coordinate folded into one unsigned integer whose
/// unsigned order is the pair's order: the value in the key's [`ValueOrder`],
/// and the lower coordinate where two values are equal.
///
/// An unpacked pair pays its tie-break and its coordinate swap on every element;
/// the key pays neither, because both ride the single comparison the value
/// already needed.
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

/// Whether an accumulation type packs into an [`OrderKey`] on this device.
///
/// The key needs the whole value beside a `u32` coordinate, so a wider
/// accumulation element has nowhere to go, and a backend without 64-bit integer
/// arithmetic (WGSL) cannot compare one at all.
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

/// The key of a slot that has taken no candidate.
///
/// It spells the last-ranked value at `u32::MAX` that the unpacked accumulator
/// starts from, so a row with nothing to rank reports the same value and index
/// either way.
#[cube]
pub(crate) fn empty_order_key<N: Numeric, S: Size>(
    #[comptime] order: ValueOrder,
) -> Vector<OrderKey, S> {
    let last = match comptime!(order) {
        ValueOrder::Descending => N::min_value(),
        ValueOrder::Ascending => N::max_value(),
    };

    pack_order_key::<N, S>(Vector::new(last), Vector::new(u32::MAX), order)
}

/// The better-ranked of two keys, whichever [`ValueOrder`] built them.
#[cube]
pub(crate) fn better_order_key<S: Size>(
    current: Vector<OrderKey, S>,
    candidate: Vector<OrderKey, S>,
) -> Vector<OrderKey, S> {
    select_many(current.greater_than(&candidate), current, candidate)
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
/// Both arms compare the float against zero rather than testing its sign bit, so
/// that a NaN of either sign fails and lands above every number.
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

            match comptime!(order) {
                ValueOrder::Descending => select_many(
                    value.less_than(&zero),
                    Vector::new(u32::MAX) - bits,
                    bits | sign,
                ),
                ValueOrder::Ascending => {
                    select_many(value.greater_than(&zero), sign - bits, bits | sign)
                }
            }
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
