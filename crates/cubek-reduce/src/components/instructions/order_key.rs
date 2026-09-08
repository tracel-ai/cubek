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

/// The codec between a value and coordinate pair and the [`OrderKey`] ranking
/// them, in one [`ValueOrder`].
///
/// Building or reading a key needs the order; comparing two of them does not,
/// since a key already carries it. That is what separates the methods taking
/// `&self` here from the associated ones that do not.
#[derive(Debug, CubeType, Clone)]
pub(crate) struct OrderedKey {
    #[cube(comptime)]
    order: ValueOrder,
}

#[cube]
impl OrderedKey {
    /// Ranks the largest value first, as top-k and max rank.
    pub fn descending() -> OrderedKey {
        OrderedKey {
            order: ValueOrder::Descending,
        }
    }

    /// Ranks the smallest value first, as min ranks.
    pub fn ascending() -> OrderedKey {
        OrderedKey {
            order: ValueOrder::Ascending,
        }
    }

    /// Whether a coordinate-tracking reduction packs into a key on this device.
    pub fn packs<P: ReducePrecision>(#[comptime] output: ReduceOutputMode) -> comptime_type!(bool) {
        let tracks_coordinates = comptime!(output.has_indices());
        let packs = OrderedKey::packs_into::<P::EA>();

        comptime!(tracks_coordinates && packs)
    }

    /// Whether an accumulation type fits: it must leave room for a `u32`
    /// coordinate beside it, and the device must be able to compare the
    /// resulting 64-bit integer (WGSL cannot).
    fn packs_into<N: Numeric>() -> comptime_type!(bool) {
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

    pub fn pack<N: Numeric, S: Size>(
        &self,
        value: Vector<N, S>,
        coordinate: Vector<u32, S>,
    ) -> Vector<OrderKey, S> {
        // Inverted, so that a lower coordinate makes a larger key and wins a tie.
        let rank = Vector::new(u32::MAX) - coordinate;

        (Vector::<OrderKey, S>::cast_from(self.order_bits::<N, S>(value)) << Vector::new(32u64))
            | Vector::<OrderKey, S>::cast_from(rank)
    }

    /// The key of a slot that has taken no candidate: `last`, the value the
    /// unpacked accumulator starts from, at coordinate `u32::MAX`, so a row with
    /// nothing to rank reports the same value and index either way.
    ///
    /// That coordinate's rank is zero, and it is written as zero rather than
    /// computed: the subtraction of a constant from itself is folded by the WGSL
    /// optimizer into a constant it cannot build for a vector type.
    pub fn empty<N: Numeric, S: Size>(&self, last: Vector<N, S>) -> Vector<OrderKey, S> {
        Vector::<OrderKey, S>::cast_from(self.order_bits::<N, S>(last)) << Vector::new(32u64)
    }

    pub fn value<N: Numeric, S: Size>(&self, key: Vector<OrderKey, S>) -> Vector<N, S> {
        self.value_from_order_bits::<N, S>(Vector::cast_from(key >> Vector::new(32u64)))
    }

    pub fn coordinate<S: Size>(key: Vector<OrderKey, S>) -> Vector<u32, S> {
        Vector::new(u32::MAX) - Vector::cast_from(key & Vector::new(0xFFFF_FFFFu64))
    }

    /// Replace a single-slot accumulator's key with whichever of it and
    /// `candidate` ranks better.
    pub fn insert<S: Size>(keys: &mut Value<Vector<OrderKey, S>>, candidate: Vector<OrderKey, S>) {
        let winning = OrderedKey::better::<S>(keys.item(), candidate);
        keys.assign(&Value::new_single(winning));
    }

    /// Insert a candidate into `keys`, held in ranked order, dropping the last.
    ///
    /// The ordering and the tie-break ride one comparison, and one selected pair
    /// swaps both halves of a slot. Only a caller that is itself unrolled `k`
    /// times asks for `unrolled`, since the budget it is priced against is the
    /// `k * k` of that nest rather than this loop's `k`.
    pub fn insert_ranked<S: Size>(
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
    pub fn finalize<S: Size>(keys: Vector<OrderKey, S>) -> OrderKey {
        let vector_size = keys.vector_size().comptime();
        let mut winning = keys.extract(0usize);

        #[unroll]
        for k in 1..vector_size {
            let candidate = keys.extract(k);
            winning = select(winning > candidate, winning, candidate);
        }

        winning
    }

    fn better<S: Size>(
        current: Vector<OrderKey, S>,
        candidate: Vector<OrderKey, S>,
    ) -> Vector<OrderKey, S> {
        select_many(current.greater_than(&candidate), current, candidate)
    }

    /// The value's bits mapped so that unsigned comparison of the results ranks
    /// the values in this order.
    ///
    /// A float's sign bit orders backwards and its magnitude bits invert under
    /// it, hence the flip. `-0.0` is mapped onto `+0.0`, since the two compare
    /// equal and a key that told them apart would break the tie towards the
    /// wrong coordinate. Every NaN takes the top key whatever its payload or
    /// sign, so NaNs outrank every number and tie among themselves, leaving the
    /// coordinate to decide as the instructions' policy states. A winning NaN
    /// therefore reads back as a canonical NaN, not as its input bits.
    fn order_bits<N: Numeric, S: Size>(&self, value: Vector<N, S>) -> Vector<u32, S> {
        let bits = Vector::<u32, S>::reinterpret(value);
        let sign = Vector::new(SIGN);
        let elem = elem_type_of::<N>();

        match comptime!(elem) {
            ElemType::Float(_) => {
                let zero = Vector::new(N::from_int(0));

                let ordered = match comptime!(self.order) {
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
            ElemType::Int(_) => self.reversed_if_ascending::<S>(bits ^ sign),
            ElemType::UInt(_) => self.reversed_if_ascending::<S>(bits),
            _ => panic!("an order key packs floats, signed and unsigned integers only"),
        }
    }

    fn value_from_order_bits<N: Numeric, S: Size>(&self, bits: Vector<u32, S>) -> Vector<N, S> {
        let sign = Vector::new(SIGN);
        let elem = elem_type_of::<N>();

        let value_bits = match comptime!(elem) {
            ElemType::Float(_) => match comptime!(self.order) {
                ValueOrder::Descending => select_many(
                    (bits & sign).equal(&Vector::new(0u32)),
                    Vector::new(u32::MAX) - bits,
                    bits ^ sign,
                ),
                // `<=` rather than a sign test, so that the one key both zeros share
                // reads back as `+0.0` here as it does in the descending arm.
                ValueOrder::Ascending => select_many(bits.less_equal(&sign), sign - bits, bits),
            },
            ElemType::Int(_) => self.reversed_if_ascending::<S>(bits) ^ sign,
            ElemType::UInt(_) => self.reversed_if_ascending::<S>(bits),
            _ => panic!("an order key packs floats, signed and unsigned integers only"),
        };

        Vector::<N, S>::reinterpret(value_bits)
    }

    /// Reverse an unsigned image that already rises with the value, so that it
    /// falls with it instead. Its own inverse, so both directions of the map use it.
    fn reversed_if_ascending<S: Size>(&self, rising: Vector<u32, S>) -> Vector<u32, S> {
        match comptime!(self.order) {
            ValueOrder::Descending => rising,
            ValueOrder::Ascending => Vector::new(u32::MAX) - rising,
        }
    }
}
