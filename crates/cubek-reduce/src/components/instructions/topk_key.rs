use cubecl::features::TypeUsage;
use cubecl::prelude::*;

/// A top-k candidate's value and coordinate folded into one unsigned integer
/// whose unsigned order is the pair's order: value descending, and coordinate
/// ascending where two values are equal.
///
/// The pair costs three comparisons and five selects per accumulator slot,
/// against one comparison and two selects for the key, because the tie-break
/// and the coordinate ride the single comparison the value already needed.
pub(crate) type TopKKey = u64;

const SIGN: u32 = 0x8000_0000;

/// Whether an accumulation type packs into a [`TopKKey`] on this device.
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
pub(crate) fn pack_topk_key<N: Numeric, S: Size>(
    value: Vector<N, S>,
    coordinate: Vector<u32, S>,
) -> Vector<TopKKey, S> {
    // Descending, so that a lower coordinate makes a larger key and wins a tie.
    let descending = Vector::new(u32::MAX) - coordinate;

    (Vector::<TopKKey, S>::cast_from(order_bits::<N, S>(value)) << Vector::new(32u64))
        | Vector::<TopKKey, S>::cast_from(descending)
}

/// The key of a slot that has taken no candidate.
///
/// It spells the `(min_value, u32::MAX)` the unpacked accumulator starts from,
/// so a row shorter than `k` reports the same value and index either way.
#[cube]
pub(crate) fn empty_topk_key<N: Numeric, S: Size>() -> Vector<TopKKey, S> {
    pack_topk_key::<N, S>(Vector::new(N::min_value()), Vector::new(u32::MAX))
}

#[cube]
pub(crate) fn topk_key_value<N: Numeric, S: Size>(key: Vector<TopKKey, S>) -> Vector<N, S> {
    value_from_order_bits::<N, S>(Vector::cast_from(key >> Vector::new(32u64)))
}

#[cube]
pub(crate) fn topk_key_coordinate<S: Size>(key: Vector<TopKKey, S>) -> Vector<u32, S> {
    Vector::new(u32::MAX) - Vector::cast_from(key & Vector::new(0xFFFF_FFFFu64))
}

/// The value's bits mapped so that unsigned comparison of the results is the
/// value's own comparison.
///
/// A float's sign bit orders backwards and its magnitude bits invert under it,
/// hence the flip. `-0.0` is mapped onto `+0.0`, since the two compare equal and
/// a key that told them apart would break the tie towards the wrong coordinate.
/// NaN has no order to preserve and lands above every finite value.
#[cube]
fn order_bits<N: Numeric, S: Size>(value: Vector<N, S>) -> Vector<u32, S> {
    let bits = Vector::<u32, S>::reinterpret(value);
    let sign = Vector::new(SIGN);
    let elem = elem_type_of::<N>();

    match comptime!(elem) {
        ElemType::Float(_) => select_many(
            value.less_than(&Vector::new(N::from_int(0))),
            Vector::new(u32::MAX) - bits,
            bits | sign,
        ),
        ElemType::Int(_) => bits ^ sign,
        ElemType::UInt(_) => bits,
        _ => panic!("a top-k key packs floats, signed and unsigned integers only"),
    }
}

#[cube]
fn value_from_order_bits<N: Numeric, S: Size>(bits: Vector<u32, S>) -> Vector<N, S> {
    let sign = Vector::new(SIGN);
    let elem = elem_type_of::<N>();

    let value_bits = match comptime!(elem) {
        ElemType::Float(_) => select_many(
            (bits & sign).equal(&Vector::new(0u32)),
            Vector::new(u32::MAX) - bits,
            bits ^ sign,
        ),
        ElemType::Int(_) => bits ^ sign,
        ElemType::UInt(_) => bits,
        _ => panic!("a top-k key packs floats, signed and unsigned integers only"),
    };

    Vector::<N, S>::reinterpret(value_bits)
}
