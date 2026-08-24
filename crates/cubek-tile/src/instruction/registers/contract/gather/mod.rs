//! N-D contraction dispatch for multiple contracted axes, projected operands, and procedural
//! filters. The execution schedules live separately because their opposite loop orders are their
//! principal performance invariant.

mod coords;
mod nd;
mod separable;

use cubecl::prelude::*;

use crate::*;

/// N-D variant of [`direct::contract`](super::direct::contract) for operations with
/// multiple contracted axes or projected operands.
///
/// The outer product it runs is an optimization, not the definition of the contraction: it holds
/// only while the lhs is free of the accumulator's column and the rhs free of its row, which is
/// what lets one read of each serve a whole row or column of cells. A resampling operand breaks
/// that: the gather makes the value depend on the output position, and a procedural weight
/// depends on it too, so an operand spanning the other's free axis is read at the cell instead,
/// and the rank-1 update degenerates to a per-cell `fma`. Only the reads change; what is
/// contracted, and the accumulator block living in registers across `kc`, do not.
///
/// A *separable* lhs takes its own schedule: one factor per contracted axis lets the weights be
/// walked in 1-D per accumulator cell rather than evaluated over their Cartesian product, which
/// is the whole cost of an expensive procedural filter.
#[cube]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] served: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] replace: bool,
) {
    let lw = lhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    let factors = lhs.factors();

    comptime!(assert!(
        factors > 0,
        "contract gather: a separable lhs must contain at least one factor"
    ));

    if comptime!(factors > 1) {
        // A separable lhs is a scalar procedural weight, so it never lines along the contracted
        // axis and its step serves one value. The block is then the accumulator's own width.
        comptime!(assert!(
            served == 1 && lw == 1,
            "contract gather: a separable lhs needs scalar weights served one value a step"
        ));
        let size!(V) = aw;
        separable::contract::<E, EL, ER, V>(acc, lhs, rhs, space, aw, config, replace);
    } else if comptime!(served > 1) {
        // The block's lines are the rhs's: `served`-wide K-partials of one cell at a folded step,
        // `aw`-wide neighbouring cells otherwise.
        let size!(W) = served;
        let size!(A) = 1usize;
        nd::nest::<E, EL, W, ER, W, A>(
            acc, lhs, rhs, space, served, lw, served, 1usize, config, replace,
        );
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nd::nest::<E, EL, W, ER, A, A>(
            acc, lhs, rhs, space, served, lw, aw, aw, config, replace,
        );
    }
}
