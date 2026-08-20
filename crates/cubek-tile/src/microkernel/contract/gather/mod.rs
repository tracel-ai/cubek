//! N-D contraction dispatch for multiple contracted axes, projected operands, and procedural
//! filters. The execution schedules live separately because their opposite loop orders are their
//! principal performance invariant.

mod coords;
mod nd;
mod separable;

use cubecl::prelude::*;

use crate::*;

use coords::assert_operand_shapes;

/// Derive the common gather problem once, validate it, then select an explicit execution schedule.
#[cube]
pub(super) fn contract<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    V: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] pack_l: usize,
    #[comptime] pack_r: usize,
    #[comptime] config: MemoryMmaConfig,
) {
    let lw = lhs.vector_size();
    let vw = rhs.vector_size();
    let size!(WPL) = comptime!(lw / pack_l);
    let size!(WPR) = comptime!(vw / pack_r);

    let rank = comptime!(space.rank());
    let (mr, nr) = comptime!((space.extent_at(rank - 2), space.extent_at(rank - 1) / vw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());

    // Contracted extents belong to the merged operands: those axes are absent from the sink.
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let reduce = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space).to_vec());
    let reduce_extents = comptime!(reduce.iter().map(|&a| merged.extent(a)).collect::<Vec<_>>());
    let kc = comptime!(reduce_extents.iter().product::<usize>());

    // These flags state exactly how far the K-major schedule may hoist each operand read.
    let lhs_spans_col = comptime!(lhs.space.contains(space.axis_at(rank - 1)));
    let rhs_spans_row = comptime!(rhs.space.contains(space.axis_at(rank - 2)));
    let rhs_spans_col = comptime!(rhs.space.contains(space.axis_at(rank - 1)));

    comptime!(assert_operand_shapes(
        &lhs.space,
        &rhs.space,
        &space,
        &reduce,
        vw,
        lhs_spans_col,
    ));

    let lhs_factors = lhs.factors();
    comptime!(assert!(
        lhs_factors > 0,
        "contract gather: a separable lhs must contain at least one factor"
    ));
    if comptime!(lhs_factors > 1) {
        comptime!(assert!(
            lhs_factors == reduce.len() && lw == 1,
            "contract gather: a separable lhs needs one factor per contracted axis and scalar \
             weights"
        ));
        separable::contract::<E, EL, ER, IR, WPR, V>(
            acc,
            lhs,
            rhs,
            comptime!(space.clone()),
            comptime!(reduce.clone()),
            comptime!(reduce_extents.clone()),
            mr,
            nr,
            vw,
            comptime!(config),
        );
    } else {
        nd::contract::<E, EL, IL, L, ER, IR, WPL, WPR, V>(
            acc,
            lhs,
            rhs,
            comptime!(space.clone()),
            comptime!(reduce.clone()),
            comptime!(reduce_extents.clone()),
            mr,
            nr,
            matrices,
            kc,
            lw,
            vw,
            lhs_spans_col,
            rhs_spans_row,
            rhs_spans_col,
            comptime!(config),
        );
    }
}
