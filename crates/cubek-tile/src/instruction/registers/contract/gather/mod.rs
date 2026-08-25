//! N-D contraction dispatch for multiple contracted axes, projected operands, and procedural
//! filters. The execution schedules live separately because their opposite loop orders are their
//! principal performance invariant.

mod coords;
mod nd;
mod separable;

use cubecl::prelude::*;

use super::shape::ContractShape;
use crate::*;

/// The gather-specific half of a contraction's comptime geometry, over the
/// [`ContractShape`] every schedule shares.
///
/// Gather coordinates are resolved many times in the generated nest. Keeping the facts that
/// define that geometry together prevents a call site from accidentally mixing spaces, reduce
/// axes, or block dimensions derived under different widths.
#[derive(Clone, Debug)]
pub(super) struct GatherProblem {
    pub block: ContractShape,
    pub lhs_space: Space,
    pub rhs_space: Space,
    /// The lhs's stated factorization, one factor per contracted axis. `None` for an lhs that
    /// answers only as a whole, which takes the general schedule.
    pub factors: Option<usize>,
    /// Factor-local normalization requested by the procedural lhs.
    pub normalization: Option<(TapMask, DivGuard)>,
    /// The separable walk's weight count: one per tap of each factor, summed rather than
    /// multiplied out.
    pub taps: usize,
    /// Where each factor's taps start in that walk.
    pub offsets: Vec<usize>,
    pub lhs_spans_col: bool,
    pub rhs_spans_row: bool,
    pub rhs_spans_col: bool,
}

impl GatherProblem {
    fn new(
        lhs: &Space,
        rhs: &Space,
        rhs_projection: &Projection,
        block: ContractShape,
        factors: Option<usize>,
        normalization: Option<(TapMask, DivGuard)>,
    ) -> Self {
        let rank = block.space.rank();
        let lhs_spans_col = lhs.contains(block.space.axis_at(rank - 1));
        let rhs_spans_row = rhs.contains(block.space.axis_at(rank - 2));
        let rhs_spans_col = rhs.contains(block.space.axis_at(rank - 1));
        coords::assert_operand_shapes(
            lhs,
            rhs,
            &block.space,
            &block.reduce,
            block.vw,
            lhs_spans_col,
        );
        if let Some(factors) = factors {
            assert_eq!(
                factors,
                block.reduce.len(),
                "contract gather: a separable lhs needs one factor per contracted axis"
            );
            coords::assert_separable_shapes(rhs_projection, &block.space, rhs_spans_col);
        }
        if normalization.is_some() {
            assert!(
                factors.is_some(),
                "contract gather: factor normalization needs a separable lhs"
            );
        }
        if matches!(normalization, Some((TapMask::Masked, _))) {
            assert_factorized_mask(
                rhs_projection,
                &block.reduce,
                block.space.axis_at(rank - 1),
                lhs_spans_col,
            );
        }
        let offsets = block
            .reduce_extents
            .iter()
            .scan(0, |start, taps| {
                let at = *start;
                *start += taps;
                Some(at)
            })
            .collect();

        Self {
            taps: block.reduce_extents.iter().sum(),
            block,
            lhs_space: lhs.clone(),
            rhs_space: rhs.clone(),
            offsets,
            factors,
            normalization,
            lhs_spans_col,
            rhs_spans_row,
            rhs_spans_col,
        }
    }
}

/// A rectangular source mask factorizes only when no physical input axis is moved by two
/// contracted axes. Output axes may share the same physical axis: they are fixed for one cached
/// tap walk and therefore do not couple factor sums.
fn assert_factorized_mask(rhs: &Projection, reduce: &[Axis], acc_col: Axis, lhs_spans_col: bool) {
    for (f, &axis) in reduce.iter().enumerate() {
        if !rhs.logical_axes().contains(&axis) {
            continue;
        }
        for pa in rhs.carriers(axis) {
            assert!(
                lhs_spans_col
                    || !rhs
                        .physical_axis(pa)
                        .terms()
                        .iter()
                        .any(|term| term.axis == acc_col),
                "contract gather: TapMask::Masked cannot cache weights across {acc_col:?} when \
                 that axis shares a source coordinate with contracted axis {axis:?}"
            );
            for &other in &reduce[(f + 1)..] {
                assert!(
                    !rhs.physical_axis(pa)
                        .terms()
                        .iter()
                        .any(|term| term.axis == other),
                    "contract gather: TapMask::Masked needs each contracted axis to move distinct \
                     input axes; {axis:?} and {other:?} both move physical axis {pa}"
                );
            }
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
/// is the whole cost of an expensive procedural filter. Its rank is the recipe's, so a stated
/// factorization of one factor takes it too: the per-row walk it caches is worth `nr` weight
/// evaluations whatever the rank.
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
    let normalization = lhs.factor_normalization();
    let rhs_projection = rhs.projection();

    let problem = comptime!(GatherProblem::new(
        &lhs.space,
        &rhs.space,
        &rhs_projection,
        ContractShape::new(&lhs.space, &rhs.space, space, served, lw, rw, aw),
        factors,
        normalization,
    ));

    if comptime!(factors.is_some()) {
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
        nd::nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, problem, config);
    } else {
        let size!(W) = lw;
        let size!(V) = rw;
        let size!(A) = aw;
        nd::nest::<E, EL, W, ER, V, A>(acc, lhs, rhs, problem, config);
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

    fn problem(factors: Option<usize>, vw: usize) -> GatherProblem {
        let (lhs, rhs, acc) = spaces();
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, vw, vw);
        GatherProblem::new(&lhs, &rhs, &projection, block, factors, None)
    }

    #[test]
    fn problem_derives_one_consistent_gather_geometry() {
        let problem = problem(Some(2), 4);
        let block = &problem.block;

        assert_eq!(block.reduce, vec![K0, K1]);
        assert_eq!(block.reduce_extents, vec![2, 3]);
        assert_eq!(problem.offsets, vec![0, 2]);
        assert_eq!((block.kc, problem.taps), (6, 5));
        assert_eq!((block.mr, block.nr), (4, 2));
        assert_eq!(block.batch_extents(), Vec::<usize>::new());
        assert_eq!(block.matrices(), 1);
        // `mr * nr` lines of `served * aw`.
        assert_eq!(block.scalars(), 32);
    }

    #[test]
    #[should_panic(expected = "one factor per contracted axis")]
    fn problem_rejects_a_factor_count_that_does_not_match_the_reduction() {
        problem(Some(3), 1);
    }

    /// The count is checked whatever it is: a rank-one factorization against a two-axis reduction
    /// is the same mismatch, and it now reaches the separable schedule rather than falling back.
    #[test]
    #[should_panic(expected = "one factor per contracted axis")]
    fn problem_rejects_a_rank_one_factorization_of_a_two_axis_reduction() {
        problem(Some(1), 1);
    }

    /// An unfactorized lhs states no factor count, so the reduction's rank is unconstrained.
    #[test]
    fn problem_accepts_an_unfactorized_lhs() {
        assert_eq!(problem(None, 4).factors, None);
    }
}
