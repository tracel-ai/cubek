//! N-D contraction dispatch for multiple contracted axes, projected operands, and procedural
//! filters. The execution schedules live separately because their opposite loop orders are their
//! principal performance invariant.

mod coords;
mod nd;
mod separable;

use cubecl::prelude::*;

use super::shape::ContractShape;
use crate::*;

/// How the lhs varies over the accumulator's axes, which is what decides how much one read of
/// it covers.
///
/// The outer product the nest is written around holds only in the first case; the other two are
/// what a gathered or batched operand degenerates to, and they differ in whether one read still
/// covers a whole cell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum LhsRole {
    /// Free of the accumulator's innermost axis, so one read serves every cell of a row.
    FreeOfColumn,
    /// Lined *along* that axis: the line it reads is the cell, every lane a different column,
    /// and there is no `K` component to extract. What a batched contraction needs -- an axis
    /// every operand spans, like a depthwise convolution's channel.
    LinedAlongColumn,
    /// Spans that axis without lining along it, so the value differs cell by cell and the
    /// rank-1 update degenerates to a per-cell `fma`.
    PerCell,
}

/// The same for the rhs, over the row the outer product assumes it is free of.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum RhsRole {
    /// Free of the row: its `nr` lines are read once per step and reused down every row.
    FreeOfRow,
    /// Varies down the rows, but still holds for a whole row of cells.
    PerRow,
    /// Varies cell by cell.
    PerCell,
}

/// The accumulator scope at which one factor's complete tap walk is computed and cached.
///
/// The same question a memory-backed operand answers through
/// [`Tile::invariant_over`](crate::Tile): the axes a value does not vary over are the ones a
/// single read of it serves, and so how far out of the nest that read lifts. A factor answers it
/// through its recipe rather than a projection, because it reads its axes instead of addressing
/// them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum FactorReuse {
    /// Once for the entire accumulator block.
    Block,
    /// Once for each accumulator row.
    Row,
    /// Once for each accumulator column.
    Column,
    /// Once for each accumulator cell.
    Cell,
}

impl FactorReuse {
    /// The scope a factor varying over these two axes can be cached at.
    pub(super) fn of(varies_row: bool, varies_col: bool) -> Self {
        match (varies_row, varies_col) {
            (false, false) => FactorReuse::Block,
            (true, false) => FactorReuse::Row,
            (false, true) => FactorReuse::Column,
            (true, true) => FactorReuse::Cell,
        }
    }
}

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
    /// Maximal reuse of each factor's tap walk within one accumulator block.
    pub factor_reuse: Vec<FactorReuse>,
    /// How each operand varies over the accumulator's own axes.
    pub lhs: LhsRole,
    pub rhs: RhsRole,
}

impl GatherProblem {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lhs: &Space,
        rhs: &Space,
        rhs_projection: &Projection,
        block: ContractShape,
        factors: Option<usize>,
        factor_dependencies: Option<Vec<(bool, bool)>>,
        normalization: Option<(TapMask, DivGuard, Space)>,
        rhs_boundaries: &[Option<Boundary>],
    ) -> Self {
        let rank = block.space.rank();
        let col = block.space.axis_at(rank - 1);
        // A separable lhs is never col-lined however it states its axes: its factors answer one
        // scalar at a time, so no read of it covers a cell, and it takes the schedule below that
        // evaluates the weights per cell.
        let lhs_role = match lhs.contains(col) {
            false => LhsRole::FreeOfColumn,
            true if factors.is_none() && lhs.axis_at(lhs.rank() - 1) == col => {
                LhsRole::LinedAlongColumn
            }
            true => LhsRole::PerCell,
        };
        let rhs_role = match rhs.contains(block.space.axis_at(rank - 2)) {
            false => RhsRole::FreeOfRow,
            true if rhs.contains(col) => RhsRole::PerCell,
            true => RhsRole::PerRow,
        };
        coords::assert_operand_shapes(
            lhs,
            rhs,
            &block.space,
            &block.reduce,
            block.lw,
            block.vw,
            lhs_role,
        );
        if let Some(factors) = factors {
            assert_eq!(
                factors,
                block.reduce.len(),
                "contract gather: a separable lhs needs one factor per contracted axis"
            );
            coords::assert_separable_shapes(rhs_projection, &block.space, rhs.contains(col));
        }
        if normalization.is_some() {
            assert!(
                factors.is_some(),
                "contract gather: factor normalization needs a separable lhs"
            );
        }
        let normalization = normalization.map(|(mask, guard, original)| {
            validate_guard(guard);
            for &axis in &block.reduce {
                assert!(
                    original.contains(axis) && original.extent_raw(axis) == lhs.extent_raw(axis),
                    "contract gather: a normalized factor axis cannot be partitioned between \
                     .normalized() and the gather leaf; calling .normalized() below a split \
                     normalizes each chunk independently"
                );
            }
            (mask, guard)
        });
        if matches!(normalization, Some((TapMask::Masked, _))) {
            assert_factorized_mask(rhs_projection, &block.reduce);
        }
        let row = block.space.axis_at(rank - 2);
        let factor_reuse = match (factors, factor_dependencies) {
            (Some(factors), Some(dependencies)) => {
                assert_eq!(dependencies.len(), factors);
                dependencies
                    .into_iter()
                    .enumerate()
                    .map(|(f, (mut varies_row, mut varies_col))| {
                        if matches!(normalization, Some((TapMask::Masked, _))) {
                            let tap = block.reduce[f];
                            varies_row |=
                                masked_bound_depends_on(rhs_projection, rhs_boundaries, tap, row);
                            varies_col |=
                                masked_bound_depends_on(rhs_projection, rhs_boundaries, tap, col);
                        }
                        FactorReuse::of(varies_row, varies_col)
                    })
                    .collect()
            }
            (None, None) => Vec::new(),
            _ => panic!("contract gather: factor dependencies must accompany the factorization"),
        };
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
            factor_reuse,
            factors,
            normalization,
            lhs: lhs_role,
            rhs: rhs_role,
        }
    }
}

/// A rectangular source mask factorizes only when no physical input axis is moved by two
/// contracted axes. Output axes may share the same physical axis: they are fixed for one cached
/// tap walk and therefore do not couple factor sums.
fn assert_factorized_mask(rhs: &Projection, reduce: &[Axis]) {
    for (f, &axis) in reduce.iter().enumerate() {
        if !rhs.logical_axes().contains(&axis) {
            continue;
        }
        for pa in rhs.carriers(axis) {
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

/// Whether the physical bound tested for `tap` also moves with accumulator `axis`.
fn masked_bound_depends_on(
    rhs: &Projection,
    boundaries: &[Option<Boundary>],
    tap: Axis,
    axis: Axis,
) -> bool {
    rhs.logical_axes().contains(&tap)
        && rhs.carriers(tap).iter().any(|&pa| {
            boundaries.get(pa).copied().flatten() == Some(Boundary::Zero)
                && rhs
                    .physical_axis(pa)
                    .terms()
                    .iter()
                    .any(|term| term.axis == axis)
        })
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
    #[comptime] contracted_per_step: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    // `step_served` only returns 1 for an rhs lining along the accumulator, where it already
    // refused anything but a matched pair or a scalar sink, so the division is exact.
    comptime!(assert!(
        rw == aw || aw == 1,
        "contract gather: a rhs staged wider than its sink spreads its lanes across scalar cells, \
         so the accumulator must be served scalar (rhs {rw}, accumulator {aw})"
    ));
    let factors = lhs.factors();
    let normalization = lhs.factor_normalization();
    let rhs_projection = rhs.projection();
    let rhs_boundaries = rhs.separable_boundaries();
    let rank = comptime!(space.rank());
    let factor_dependencies = lhs.factor_dependencies(
        factors,
        comptime!(space.axis_at(rank - 2)),
        comptime!(space.axis_at(rank - 1)),
    );

    let problem = comptime!(GatherProblem::new(
        &lhs.space,
        &rhs.space,
        &rhs_projection,
        ContractShape::new(
            &lhs.space,
            &rhs.space,
            space,
            contracted_per_step,
            lw,
            rw,
            aw
        ),
        factors,
        factor_dependencies,
        normalization,
        &rhs_boundaries,
    ));

    if comptime!(factors.is_some()) {
        // A separable lhs is a scalar procedural weight, so it never lines along the contracted
        // axis and its step serves one value. The block is then the accumulator's own width.
        comptime!(assert!(
            contracted_per_step == 1 && lw == 1,
            "contract gather: a separable lhs needs scalar weights contracted_per_step one value a step"
        ));
        let size!(V) = rw;
        let size!(A) = aw;
        separable::contract::<E, EL, ER, V, A>(acc, lhs, rhs, problem, config, semiring);
    } else if comptime!(contracted_per_step > 1) {
        // The block's lines are the rhs's: `contracted_per_step`-wide K-partials of one cell at a folded step,
        // `aw`-wide neighbouring cells otherwise.
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        nd::nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, problem, config, semiring);
    } else {
        let size!(W) = lw;
        let size!(V) = rw;
        let size!(A) = aw;
        nd::nest::<E, EL, W, ER, V, A>(acc, lhs, rhs, problem, config, semiring);
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
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            factors,
            factors.map(|n| vec![(true, true); n]),
            None,
            &[],
        )
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
        // `mr * nr` lines of `contracted_per_step * aw`.
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

    #[test]
    #[should_panic(expected = "normalized factor axis cannot be partitioned")]
    fn problem_rejects_chunk_local_factor_normalization() {
        let (lhs, rhs, acc) = spaces();
        let original = Space::new(&[(M, 4), (K0, 4), (K1, 3)]);
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(true, true); 2]),
            Some((TapMask::Unmasked, DivGuard::default(), original)),
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "needs each contracted axis to move distinct input axes")]
    fn problem_rejects_masked_normalization_when_contracted_axes_share_input_axis() {
        let (lhs, rhs, acc) = spaces();
        let map = vec![
            PhysicalAxisMap::affine(&[(K0, 1), (K1, 1)]),
            PhysicalAxisMap::of(N),
        ];
        let projection = Projection::new(&[K0, K1, N], &map);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(true, true); 2]),
            Some((TapMask::Masked, DivGuard::default(), lhs.clone())),
            &[],
        );
    }

    #[test]
    fn a_masked_bound_blocks_only_the_unsafe_column_hoist() {
        let (lhs, rhs, acc) = spaces();
        // K0's checked carrier also moves with N. K1 has its own carrier, while the final N
        // carrier satisfies the separable reader's innermost-axis requirement.
        let map = vec![
            PhysicalAxisMap::affine(&[(K0, 1), (N, 1)]),
            PhysicalAxisMap::of(K1),
            PhysicalAxisMap::of(N),
        ];
        let projection = Projection::new(&[K0, K1, N], &map);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        let problem = GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(false, false); 2]),
            Some((TapMask::Masked, DivGuard::default(), lhs.clone())),
            &[Some(Boundary::Zero), None, None],
        );

        assert_eq!(problem.factor_reuse[0], FactorReuse::Column);
        assert_eq!(problem.factor_reuse[1], FactorReuse::Block);
    }

    #[test]
    fn recipe_dependencies_choose_the_maximal_factor_cache() {
        let (lhs, rhs, acc) = spaces();
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        let problem = GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(true, false), (false, true)]),
            None,
            &[],
        );

        assert_eq!(
            problem.factor_reuse,
            vec![FactorReuse::Row, FactorReuse::Column]
        );
    }

    #[test]
    fn constant_and_fully_dependent_factors_choose_the_extreme_caches() {
        let (lhs, rhs, acc) = spaces();
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        let problem = GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(false, false), (true, true)]),
            None,
            &[],
        );

        assert_eq!(
            problem.factor_reuse,
            vec![FactorReuse::Block, FactorReuse::Cell]
        );
    }

    /// An unfactorized lhs states no factor count, so the reduction's rank is unconstrained.
    #[test]
    fn problem_accepts_an_unfactorized_lhs() {
        assert_eq!(problem(None, 4).factors, None);
    }

    /// Every operand spanning one axis, with the lhs lining along it, is what a batched
    /// contraction is -- a depthwise convolution's channel. `lhs` names how the lhs orders its
    /// own axes, which is what separates lining along the column from merely spanning it.
    fn batched(lhs: &[(Axis, usize)], factors: Option<usize>, width: usize) -> GatherProblem {
        let lhs = Space::new(lhs);
        let rhs = Space::new(&[(M, 4), (K0, 2), (K1, 3), (N, 8)]);
        let acc = Space::new(&[(M, 4), (N, 8)]);
        let projection = Projection::direct(&[M, K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, width, width, width);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            factors,
            factors.map(|n| vec![(true, true); n]),
            None,
            &[],
        )
    }

    /// An lhs lined along the accumulator's column reads its line *as* the cell, with no `K`
    /// component left to extract -- so it is exempt from lining along the contracted axis.
    #[test]
    fn a_col_lined_lhs_is_read_as_the_cell() {
        let problem = batched(&[(K0, 2), (K1, 3), (N, 8)], None, 4);

        assert_eq!(problem.lhs, LhsRole::LinedAlongColumn);
        assert_eq!(problem.rhs, RhsRole::PerCell);
    }

    /// Spanning the column without lining along it is the other case: the value differs cell by
    /// cell, so one read cannot cover a line of them and the accumulator has to be scalar.
    #[test]
    fn an_lhs_spanning_the_column_off_its_line_is_read_per_cell() {
        let problem = batched(&[(K0, 2), (N, 8), (K1, 3)], None, 1);

        assert_eq!(problem.lhs, LhsRole::PerCell);
    }

    /// A separable lhs answers one scalar at a time, so no read of it covers a cell and it takes
    /// the per-cell schedule whatever its axis order says.
    #[test]
    fn a_separable_lhs_is_never_col_lined() {
        let problem = batched(&[(K0, 2), (N, 8), (K1, 3)], Some(2), 1);

        assert_eq!(problem.lhs, LhsRole::PerCell);
    }
}
