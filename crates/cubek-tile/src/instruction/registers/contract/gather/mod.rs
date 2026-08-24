//! N-D contraction dispatch for multiple contracted axes, projected operands, and procedural
//! filters. The execution schedules live separately because their opposite loop orders are their
//! principal performance invariant.

mod coords;
mod nd;
mod separable;

use cubecl::prelude::*;

use crate::*;

/// Comptime geometry shared by both gathered contraction schedules.
///
/// Gather coordinates are resolved many times in the generated nest. Keeping the facts that
/// define that geometry together prevents a call site from accidentally mixing spaces, reduce
/// axes, or block dimensions derived under different widths.
#[derive(Clone, Debug)]
pub(super) struct GatherProblem {
    pub space: Space,
    pub lhs_space: Space,
    pub rhs_space: Space,
    pub reduce: Vec<Axis>,
    pub reduce_extents: Vec<usize>,
    pub factors: usize,
    pub kc: usize,
    pub taps: usize,
    pub offsets: Vec<usize>,
    pub mr: usize,
    pub nr: usize,
    pub vw: usize,
    pub lhs_spans_col: bool,
    pub rhs_spans_row: bool,
    pub rhs_spans_col: bool,
}

impl GatherProblem {
    fn new(
        lhs: &Space,
        rhs: &Space,
        space: Space,
        factors: usize,
        vw: usize,
        cell_width: usize,
    ) -> Self {
        let rank = space.rank();
        assert!(
            factors > 0,
            "contract gather: a separable lhs must contain at least one factor"
        );
        let merged = Space::merge(&[lhs, rhs]);
        let reduce = Space::contracted(&[lhs, rhs], &space).to_vec();
        let reduce_extents = reduce
            .iter()
            .map(|&axis| merged.extent(axis))
            .collect::<Vec<_>>();
        let lhs_spans_col = lhs.contains(space.axis_at(rank - 1));
        let rhs_spans_row = rhs.contains(space.axis_at(rank - 2));
        let rhs_spans_col = rhs.contains(space.axis_at(rank - 1));
        let mr = space.extent_at(rank - 2);
        let nr = space.extent_at(rank - 1) / cell_width;
        coords::assert_operand_shapes(lhs, rhs, &space, &reduce, vw, lhs_spans_col);
        if factors > 1 {
            assert_eq!(
                factors,
                reduce.len(),
                "contract gather: a separable lhs needs one factor per contracted axis"
            );
        }
        let offsets = reduce_extents
            .iter()
            .scan(0, |start, taps| {
                let at = *start;
                *start += taps;
                Some(at)
            })
            .collect();

        Self {
            space,
            lhs_space: lhs.clone(),
            rhs_space: rhs.clone(),
            kc: reduce_extents.iter().product(),
            taps: reduce_extents.iter().sum(),
            offsets,
            mr,
            nr,
            reduce,
            reduce_extents,
            factors,
            vw,
            lhs_spans_col,
            rhs_spans_row,
            rhs_spans_col,
        }
    }
}

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
) {
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    let factors = lhs.factors();

    let cell_width = comptime!(if served > 1 { 1usize } else { rw });
    let problem = comptime!(GatherProblem::new(
        &lhs.space, &rhs.space, space, factors, rw, cell_width,
    ));

    if comptime!(factors > 1) {
        // A separable lhs is a scalar procedural weight, so it never lines along the contracted
        // axis and its step serves one value. The block is then the accumulator's own width.
        comptime!(assert!(
            served == 1 && lw == 1,
            "contract gather: a separable lhs needs scalar weights served one value a step"
        ));
        let size!(V) = rw;
        separable::contract::<E, EL, ER, V>(acc, lhs, rhs, problem, config);
    } else if comptime!(served > 1) {
        // The block's lines are the rhs's: `served`-wide K-partials of one cell at a folded step,
        // `aw`-wide neighbouring cells otherwise.
        let size!(W) = served;
        let size!(A) = 1usize;
        nd::nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, problem, served, lw, 1usize, config);
    } else {
        let size!(W) = lw;
        let size!(V) = rw;
        let size!(A) = aw;
        nd::nest::<E, EL, W, ER, V, A>(acc, lhs, rhs, problem, served, lw, aw, config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K0: Axis = Axis(2);
    const K1: Axis = Axis(3);

    fn spaces() -> (Space, Space, Space) {
        (
            Space::new(&[(M, 4), (K0, 2), (K1, 3)]),
            Space::new(&[(K0, 2), (K1, 3), (N, 8)]),
            Space::new(&[(M, 4), (N, 8)]),
        )
    }

    #[test]
    fn problem_derives_one_consistent_gather_geometry() {
        let (lhs, rhs, acc) = spaces();
        let problem = GatherProblem::new(&lhs, &rhs, acc, 2, 4, 4);

        assert_eq!(problem.reduce, vec![K0, K1]);
        assert_eq!(problem.reduce_extents, vec![2, 3]);
        assert_eq!(problem.offsets, vec![0, 2]);
        assert_eq!((problem.kc, problem.taps), (6, 5));
        assert_eq!((problem.mr, problem.nr), (4, 2));
    }

    #[test]
    #[should_panic(expected = "one factor per contracted axis")]
    fn problem_rejects_a_factor_count_that_does_not_match_the_reduction() {
        let (lhs, rhs, acc) = spaces();
        GatherProblem::new(&lhs, &rhs, acc, 3, 1, 1);
    }
}
