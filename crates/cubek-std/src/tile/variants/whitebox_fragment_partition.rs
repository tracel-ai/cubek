use cubecl;
use cubecl::prelude::*;

use crate::tile::RowWise;
use crate::tile::variants::whitebox_fragment::{WhiteboxFragment, WhiteboxFragmentLayout};

/// Partition holding a [`Sequence`] of [`WhiteboxFragment`]s. The cross-plane
/// row reducer (`row_max`, `row_sum`) lives here because it conceptually
/// operates over several plane-fragmented [`WhiteboxFragment`]s at once. The
/// typical case today is `len == 1` (one fragment per partition); future
/// multi-fragment partitions will exercise the iteration semantics of the
/// methods below.
///
/// This is the first "partition as tile" variant; more partitions over other
/// per-variant tile types are expected to follow.
#[derive(CubeType)]
pub struct WhiteboxFragmentPartition<N: Numeric> {
    pub fragments: Sequence<WhiteboxFragment<N>>,
    #[cube(comptime)]
    pub num_fragments: u32,
}

#[cube]
impl<N: Numeric> WhiteboxFragmentPartition<N> {
    pub fn new(
        #[comptime] num_fragments: u32,
        #[comptime] layout: WhiteboxFragmentLayout,
    ) -> WhiteboxFragmentPartition<N> {
        let mut fragments = Sequence::<WhiteboxFragment<N>>::new();
        #[unroll]
        for _ in 0..num_fragments {
            fragments.push(WhiteboxFragment::<N>::new(layout));
        }
        WhiteboxFragmentPartition::<N> {
            fragments,
            num_fragments,
        }
    }
}

#[cube]
impl<E: Float> WhiteboxFragmentPartition<E> {
    /// Cross-plane row-max across the partition. For `num_fragments == 1` this
    /// is exactly the standard single-fragment plane reduce. For
    /// `num_fragments > 1` each fragment plane-reduces independently and the
    /// per-fragment results are max-combined locally into `acc`.
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        acc.copy_from(base);
        let n = comptime!(self.num_fragments);
        #[unroll]
        for i in 0..n {
            reduce::<E, FragmentRowMax>(acc, self.fragments.index(i as usize));
        }
    }

    /// Cross-plane row-sum across the partition. For `num_fragments > 1` each
    /// fragment plane-reduces independently and per-fragment results are
    /// added into `acc`.
    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        acc.fill(E::from_int(0));
        let n = comptime!(self.num_fragments);
        #[unroll]
        for i in 0..n {
            reduce::<E, FragmentRowSum>(acc, self.fragments.index(i as usize));
        }
    }
}

// ===========================================================================
// Single-fragment helpers
//
// The cross-plane reducer is the same code regardless of whether you go
// through a [`WhiteboxFragmentPartition`] or just have a single
// [`WhiteboxFragment`] in hand (e.g. the inner fragment of a
// [`crate::tile::variants::BounceTile`]). Exposed `pub(crate)` so single-
// fragment callers don't have to materialize a partition.
// ===========================================================================

#[cube]
pub(crate) fn fragment_row_max<E: Float>(
    fragment: &WhiteboxFragment<E>,
    acc: &mut RowWise<E>,
    base: &RowWise<E>,
) {
    acc.copy_from(base);
    reduce::<E, FragmentRowMax>(acc, fragment);
}

#[cube]
pub(crate) fn fragment_row_sum<E: Float>(fragment: &WhiteboxFragment<E>, acc: &mut RowWise<E>) {
    acc.fill(E::from_int(0));
    reduce::<E, FragmentRowSum>(acc, fragment);
}

// ===========================================================================
// Cross-plane reducer (private)
//
// Reduces row-wise quantities across plane units that share a row, masking
// out off-row peers. Restricted to plane scope (uses `plane_shuffle` and
// `UNIT_POS_X`); callers enforce that.
// ===========================================================================

#[cube]
fn reduce<E: Float, RO: ReduceOp<E>>(vals: &mut RowWise<E>, data: &WhiteboxFragment<E>) {
    let num_units_per_row = data.num_units_per_row().comptime();
    let num_shares_within_plane = num_units_per_row.next_power_of_two().ilog2();

    let unit_pos = UNIT_POS_X;
    let unit_pos_in_row = unit_pos % num_units_per_row;

    RO::reduce_local(data, vals);

    for i in 0..num_shares_within_plane {
        let offset = num_units_per_row >> (i + 1);
        let source_unit = unit_pos + offset;

        let value_from_source = rowwise_plane_broadcast(vals, source_unit);

        // Mask if outside the row
        let mask = unit_pos_in_row + offset >= num_units_per_row;
        RO::reduce_from_peer(vals, &value_from_source, mask);
    }

    // Broadcast back to subgroup
    let result = &rowwise_plane_broadcast(vals, unit_pos - unit_pos_in_row);
    vals.copy_from(result);
}

#[cube]
fn rowwise_plane_broadcast<E: Float>(rowwise: &RowWise<E>, source_unit: u32) -> RowWise<E> {
    let mut result = Array::new(rowwise.num_rows);

    for r in 0..rowwise.num_rows {
        result[r] = plane_shuffle(rowwise.vals[r], source_unit);
    }

    RowWise::<E> {
        num_rows: rowwise.num_rows,
        vals: result,
    }
}

#[cube]
trait ReduceOp<E: Float> {
    fn reduce_local(data: &WhiteboxFragment<E>, acc: &mut RowWise<E>);
    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}

#[derive(CubeType)]
struct FragmentRowMax {}

#[derive(CubeType)]
struct FragmentRowSum {}

#[cube]
impl<E: Float> ReduceOp<E> for FragmentRowMax {
    fn reduce_local(data: &WhiteboxFragment<E>, acc: &mut RowWise<E>) {
        acc.max_inplace(&data.rowwise_max())
    }

    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(mask) * E::min_value());
        masked.add_inplace(elem);

        acc.max_inplace(&masked)
    }
}

#[cube]
impl<E: Float> ReduceOp<E> for FragmentRowSum {
    fn reduce_local(data: &WhiteboxFragment<E>, acc: &mut RowWise<E>) {
        acc.add_inplace(&data.rowwise_sum())
    }

    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(!mask));
        masked.mul_inplace(elem);

        acc.add_inplace(&masked)
    }
}
