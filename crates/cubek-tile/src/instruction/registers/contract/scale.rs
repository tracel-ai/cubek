//! What a scales operand is against the values it covers.
//!
//! A scale is a tile spanning fewer axes than the values it multiplies: "one scale per block" is
//! what its axes say, not arithmetic. The fold side is stated ([`Scaling`](crate::Scaling)) and
//! checked here; a read is a lookup ([`ScaleLookup`](crate::ScaleLookup)) needing nothing here.

use crate::*;
/// Which factor of a contraction's terms an operand is: the argument position, which the caller
/// knows because it is the one filling the slots.
///
/// One verb serves both. `(a ⊗ s) · b` and `a · (b ⊗ s)` are the same sum of terms, the scale one
/// more factor of each, and which operand carries it is only *where* it folds in cheapest: once
/// per `(row, k)` beside the lhs, once per `(col, k)` beside the rhs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    /// The contraction's left factor: its scales span the accumulator's rows, or the contracted
    /// axis alone, and fold into the value before it forms its products.
    Lhs,
    /// The contraction's right factor: its scales span the accumulator's columns, or the
    /// contracted axis alone, and fold into each line.
    Rhs,
}

/// Refuse scales that do not ride the operand the kernel put them on.
///
/// Scales over the accumulator's columns are a fact about the rhs's columns and fold nowhere
/// else; over its rows, about the lhs's. Either on the other operand scales the output, not a term.
/// Scales over neither axis (per-tensor, or per block of `k`) ride whichever side was stated.
pub(crate) fn check_scales_ride(side: Side, scales: &Space, output: &Space, axes: MatrixAxes) {
    let group = |range: core::ops::Range<usize>| {
        range
            .filter(|&p| scales.contains(output.axis_at(p)))
            .map(|p| output.axis_at(p))
            .collect::<Vec<_>>()
    };
    let rows = group(axes.row_split..axes.col_split);
    let cols = group(axes.col_split..output.rank());
    let (own, other, foreign) = match side {
        Side::Lhs => ("lhs", "rhs", cols),
        Side::Rhs => ("rhs", "lhs", rows),
    };
    assert!(
        foreign.is_empty(),
        "mm_scaled: the scales ride the {own} but span {foreign:?}, which only the {other} \
         varies over; a scale over both operands' own axes is a scale of the output, not a \
         factor of either term"
    );
}

/// Refuse a scales operand that spells its granularity by dividing.
///
/// A scale covers a block because the operand has no axis to vary over inside one: the block is
/// an axis the scales omit. A rational axis (`PhysicalAxisMap::of(N).over(bn)`) states the same
/// granularity, but whether a line straddles a block then depends on the unstated line width.
pub(crate) fn check_scales_omit_rather_than_divide(scales: &Projection) {
    for pa in 0..scales.physical_rank() {
        let divisor = scales.divisor(pa).bound();
        assert!(
            divisor == 1,
            "mm_scaled: this scales operand divides a logical axis by {divisor} to reach its \
             block. Spell the block as an axis of its own and omit the position inside it, so one \
             scale per block is what the operand's axes say rather than what its arithmetic does"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    fn contraction() -> (Space, MatrixAxes) {
        let out = Space::new(&[(M, 4), (N, 4)]);
        let lhs = Space::new(&[(M, 4), (K, 8)]);
        let axes = MatrixAxes::accumulator(&out, &lhs);
        (out, axes)
    }

    /// Scales over the accumulator's rows ride the lhs; over its columns, the rhs.
    #[test]
    fn scales_ride_the_operand_whose_axis_they_span() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(M, 4)]), &out, axes);
        check_scales_ride(Side::Rhs, &Space::new(&[(N, 4)]), &out, axes);
    }

    /// A scale over no matrix axis is the same number on either side, so the statement stands.
    #[test]
    fn a_scale_over_no_axis_rides_either_side() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(K, 8)]), &out, axes);
        check_scales_ride(Side::Rhs, &Space::new(&[(K, 8)]), &out, axes);
    }

    #[test]
    #[should_panic(expected = "ride the lhs but span [Axis(1)]")]
    fn column_scales_cannot_ride_the_lhs() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(N, 4)]), &out, axes);
    }

    #[test]
    #[should_panic(expected = "ride the rhs but span [Axis(0)]")]
    fn row_scales_cannot_ride_the_rhs() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Rhs, &Space::new(&[(M, 4)]), &out, axes);
    }
}
