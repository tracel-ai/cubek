use cubecl::features::TypeUsage;
use cubecl::prelude::*;

use super::extremum::numeric_is_nan;
use crate::components::instructions::ReduceOutputMode;
use crate::components::precision::ReducePrecision;

/// A value and its coordinate folded into one unsigned integer, so that one
/// unsigned comparison ranks the pair: by value, largest first, and by the lower
/// coordinate where two values are equal.
pub(crate) type Packed = u64;

const SIGN: u32 = 0x8000_0000;

/// Packs a value and its coordinate into a [`Packed`], and reads them back.
///
/// Only the largest-first order exists, because only top-k packs and top-k ranks
/// that way. Max and min rank their single slot cheaper unpacked, since packing
/// pays for itself per slot but builds its value per element, so nothing asks for
/// the reverse map.
pub(crate) struct Packing {}

#[cube]
impl Packing {
    /// Whether a coordinate-tracking reduction packs on this device.
    pub fn packs<P: ReducePrecision>(#[comptime] output: ReduceOutputMode) -> comptime_type!(bool) {
        let tracks_coordinates = comptime!(output.has_indices());
        let packs = Packing::packs_into::<P::EA>();

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
        value: Vector<N, S>,
        coordinate: Vector<u32, S>,
    ) -> Vector<Packed, S> {
        // Inverted, so that a lower coordinate makes a larger packed and wins a tie.
        let tie_break = Vector::new(u32::MAX) - coordinate;

        Packing::fold::<S>(Packing::order_bits::<N, S>(value), tie_break)
    }

    fn fold<S: Size>(rank: Vector<u32, S>, tie_break: Vector<u32, S>) -> Vector<Packed, S> {
        (Vector::<Packed, S>::cast_from(rank) << Vector::new(32u64))
            | Vector::<Packed, S>::cast_from(tie_break)
    }

    /// Read back the ranking half, to weigh a candidate against a slot without
    /// unpacking either into a value.
    pub fn rank_of<S: Size>(packed: Vector<Packed, S>) -> Vector<u32, S> {
        Vector::cast_from(packed >> Vector::new(32u64))
    }

    /// Whether any lane holds a rank that reaches `threshold`.
    ///
    /// The lanes rank independently, so one lane wanting to insert is enough to
    /// make the whole vector do the work. Summing is how a vector answers this,
    /// since the only reduction cubecl exposes over lanes is a sum.
    ///
    /// Equal ranks reach it rather than fall short of it, because a slot still
    /// holding [`Self::empty`] parks its coordinate at `u32::MAX` and loses the
    /// tie it would win at any real coordinate. Ranking a candidate that turns
    /// out not to displace anything costs a walk; dropping one that would have
    /// leaves the identity's coordinate in the output.
    pub fn any_reaching<S: Size>(rank: Vector<u32, S>, threshold: Vector<u32, S>) -> bool {
        let reaches = rank.greater_equal(&threshold);

        Vector::<u32, S>::cast_from(reaches).vector_sum() != 0
    }

    /// What a slot that has taken nothing holds: `last`, the value the unpacked
    /// accumulator starts from, at coordinate `u32::MAX`, so a row with nothing
    /// to rank reports the same value and index either way.
    ///
    /// That coordinate's rank is zero, and it is written as zero rather than
    /// computed: the subtraction of a constant from itself is folded by the WGSL
    /// optimizer into a constant it cannot build for a vector type.
    pub fn empty<N: Numeric, S: Size>(last: Vector<N, S>) -> Vector<Packed, S> {
        Vector::<Packed, S>::cast_from(Packing::order_bits::<N, S>(last)) << Vector::new(32u64)
    }

    pub fn value<N: Numeric, S: Size>(packed: Vector<Packed, S>) -> Vector<N, S> {
        Packing::value_from_order_bits::<N, S>(Vector::cast_from(packed >> Vector::new(32u64)))
    }

    pub fn coordinate<S: Size>(packed: Vector<Packed, S>) -> Vector<u32, S> {
        Vector::new(u32::MAX) - Vector::cast_from(packed & Vector::new(0xFFFF_FFFFu64))
    }

    /// Insert a candidate into `packed`, held in ranked order, dropping the last.
    ///
    /// The ordering and the tie-break ride one comparison, and one selected pair
    /// swaps both halves of a slot. Only a caller that is itself unrolled `k`
    /// times asks for `unrolled`, since the budget it is priced against is the
    /// `k * k` of that nest rather than this loop's `k`.
    pub fn insert_ranked<S: Size>(
        packed: &mut Array<Vector<Packed, S>>,
        insert: Vector<Packed, S>,
        #[comptime] k: usize,
        #[comptime] unrolled: bool,
    ) {
        let mut insert = insert;

        #[unroll(unrolled)]
        for j in 0..k {
            let to_keep = packed[j].greater_than(&insert);
            let next = select_many(to_keep, insert, packed[j]);
            packed[j] = select_many(to_keep, packed[j], insert);
            insert = next;
        }
    }

    /// The value's bits mapped so that unsigned comparison of the results ranks
    /// the values largest first.
    ///
    /// A float's sign bit orders backwards and its magnitude bits invert under
    /// it, hence the flip. `-0.0` is mapped onto `+0.0`, since the two compare
    /// equal and a packing that told them apart would break the tie towards the
    /// wrong coordinate. Every NaN packs to the top whatever its payload or
    /// sign, so NaNs outrank every number and tie among themselves, leaving the
    /// coordinate to decide as the instructions' policy states. A winning NaN
    /// therefore reads back as a canonical NaN, not as its input bits.
    fn order_bits<N: Numeric, S: Size>(value: Vector<N, S>) -> Vector<u32, S> {
        Packing::order_bits_of::<N, S>(value, true)
    }

    /// [`Self::order_bits`] without the NaN fix-up, which a threshold does not
    /// need: every NaN already maps above every number, since its exponent is
    /// all ones and this map only ever flips the sign bit. What the fix-up adds
    /// is that NaNs tie *with each other*, and a threshold only asks which side
    /// of it a candidate falls on. A NaN weighed against a slot already holding
    /// one falls short of that slot's exact `u32::MAX` and is rejected, which is
    /// the same answer the tie would have given it, since the earlier
    /// coordinate wins.
    ///
    /// It is not enough to store: a slot has to tie, so an accepted candidate is
    /// ranked again exactly.
    pub fn rank_loose<N: Numeric, S: Size>(value: Vector<N, S>) -> Vector<u32, S> {
        Packing::order_bits_of::<N, S>(value, false)
    }

    fn order_bits_of<N: Numeric, S: Size>(
        value: Vector<N, S>,
        #[comptime] exact: bool,
    ) -> Vector<u32, S> {
        let bits = Vector::<u32, S>::reinterpret(value);
        let sign = Vector::new(SIGN);
        let elem = elem_type_of::<N>();

        match comptime!(elem) {
            ElemType::Float(_) => {
                let zero = Vector::new(N::from_int(0));
                let ordered = select_many(
                    value.less_than(&zero),
                    Vector::new(u32::MAX) - bits,
                    bits | sign,
                );

                if comptime!(exact) {
                    select_many(numeric_is_nan(value), Vector::new(u32::MAX), ordered)
                } else {
                    ordered
                }
            }
            ElemType::Int(_) => bits ^ sign,
            ElemType::UInt(_) => bits,
            _ => panic!("a packed value packs floats, signed and unsigned integers only"),
        }
    }

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
            _ => panic!("a packed value packs floats, signed and unsigned integers only"),
        };

        Vector::<N, S>::reinterpret(value_bits)
    }
}
