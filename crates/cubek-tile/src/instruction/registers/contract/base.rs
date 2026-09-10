//! The contraction nest's entry point: settle how many contracted values a step consumes, then
//! route to the 2-D or the N-D nest.

use cubecl::prelude::*;

use super::direct;
use super::gather;
use super::shape::ContractShape;
use crate::*;

/// Run the register instruction over each batch matrix, reading operands through the
/// quant-transparent [`matrix_packed`](Tile::matrix_packed). Each factor resolves its own
/// [`Packing`], so neither side constrains the other's, and each carries its own scales.
///
/// The 2-D nest reads each operand as a batch matrix, which describes it only when one axis is
/// contracted *and* a logical coordinate is a physical one. Either condition failing takes the
/// N-D nest, so a stencil contracting a single axis is a gather just as much as a two-axis
/// reduce is. A scaled factor takes the 2-D nest alone: the N-D nest reads its operands through
/// compacted gather windows, where a step has no single scalar `k` to address a scale with, and
/// that is a second design question rather than a second copy of this one.
#[cube]
pub(crate) fn memory<E: Numeric, EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Scaled<EL, LS>,
    rhs: &Scaled<ER, RS>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_values = lhs.values();
    let rhs_values = rhs.values();
    let lhs_levels = lhs.levels();
    let rhs_levels = rhs.levels();
    let lhs_count = lhs_levels.len();
    let rhs_count = rhs_levels.len();
    let scaled = comptime!(lhs_count > 0 || rhs_count > 0);

    let lhs_gathered = lhs_values.gathered();
    let rhs_gathered = rhs_values.gathered();
    let lhs_procedural = lhs_values.is_procedural();
    let rhs_procedural = rhs_values.is_procedural();
    let lw = lhs_values.vector_size();
    let rw = rhs_values.vector_size();
    let aw = comptime!(acc.store.vector_size);
    let contracted_per_step = comptime!(contracted_per_step(
        &lhs_values.space,
        &rhs_values.space,
        &space,
        lw,
        rw,
        aw
    ));
    // Whether a 2-D reading describes the operands is the operands' own answer, not an axis count:
    // several contracted axes still form one `k` edge when the operand carries them as one run,
    // which is what a partitioned axis is.
    let shape = comptime!(ContractShape::new(
        &lhs_values.space,
        &rhs_values.space,
        space.clone(),
        contracted_per_step,
        lw,
        rw,
        aw
    ));
    let flat = comptime!(
        shape
            .matrix_axes(&lhs_values.space, &rhs_values.space)
            .is_some()
    );
    let nd = comptime!(
        !flat
            || lhs_gathered
            || rhs_gathered
            || lhs_procedural
            || rhs_procedural
            || (contracted_per_step == 1 && rw != aw)
    );
    comptime!(assert!(
        !(nd && scaled),
        "mm: a scaled factor reads as one matrix. This contraction needs the N-D nest, which \
         addresses every operand at the cell instead"
    ));

    if nd {
        gather::contract::<E, EL, ER>(
            acc,
            &lhs_values,
            &rhs_values,
            space,
            contracted_per_step,
            config,
            semiring,
        );
    } else {
        direct::contract::<E, EL, LS, ER, RS>(
            acc,
            &lhs_values,
            &lhs_levels,
            &rhs_values,
            &rhs_levels,
            space,
            contracted_per_step,
            config,
            semiring,
        );
    }
}

/// How many contracted values one step consumes, reconciled across both operands and the
/// accumulator: one where the rhs lines along the accumulator, its whole line where it lines
/// along the contraction, so a block's lines are then partials of one cell.
///
/// Asked per operand ([`Space::contracted_per_step`]) because the answer differs per operand: an lhs lined
/// along the contracted axis folds, an rhs lined along the accumulator's innermost axis holds
/// cells that must stay apart. Both must serve the same count, and the block's lanes mean one
/// axis, so a folded step needs a scalar-contracted_per_step accumulator. Settled once per block,
/// whether the block is the memory leaf's or opened ahead of the walk
/// ([`Tile::block_accumulator`]).
pub(crate) fn contracted_per_step(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    lw: usize,
    rw: usize,
    aw: usize,
) -> usize {
    let contracted = Space::contracted(&[lhs, rhs], acc);
    let k = contracted[contracted.len() - 1];
    let lined = rhs.axis_at(rhs.rank() - 1);
    if lined != k {
        assert!(
            rw == aw || aw == 1,
            "contract: the rhs lines along {lined:?} together with the accumulator, so both must \
             be served at one width unless the accumulator is scalar and the rhs is a padded \
             stage (rhs {rw}, accumulator {aw})"
        );
        return 1;
    }
    let contracted_per_step = rhs.contracted_per_step(&contracted, rw);
    assert!(
        contracted_per_step > 1,
        "contract: the rhs lines along the contracted axis {k:?}, which is served in whole lines; \
         its width {rw} must exceed 1 and divide the axis's extent {}",
        rhs.extent(k)
    );
    assert_eq!(
        lhs.contracted_per_step(&contracted, lw),
        contracted_per_step,
        "contract: the rhs serves {contracted_per_step} contracted values a step; line the lhs along {k:?} at \
         the same width (it is {lw} wide)"
    );
    assert_eq!(
        aw, 1,
        "contract: a step serving {contracted_per_step} contracted values holds partials of one cell in the \
         block's lanes, so the accumulator cannot also be served in {aw}-wide lines"
    );
    contracted_per_step
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    fn spaces(lhs: &[Axis], rhs: &[Axis]) -> (Space, Space, Space) {
        let extents = [(M, 4usize), (N, 4), (K, 8)];
        let pick = |axes: &[Axis]| {
            Space::new(
                &axes
                    .iter()
                    .map(|&a| (a, extents.iter().find(|e| e.0 == a).unwrap().1))
                    .collect::<Vec<_>>(),
            )
        };
        (pick(lhs), pick(rhs), pick(&[M, N]))
    }

    /// An rhs lined along the accumulator holds cells that must stay apart, whatever the lhs's
    /// line width.
    #[test]
    fn an_rhs_lined_along_the_accumulator_serves_one_value_a_step() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[K, N]);
        assert_eq!(contracted_per_step(&lhs, &rhs, &acc, 4, 2, 2), 1);
    }

    /// Both operands lined along the contracted axis: the lanes are partials of one cell.
    #[test]
    fn both_operands_lined_along_the_contracted_axis_serve_a_line() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        assert_eq!(contracted_per_step(&lhs, &rhs, &acc, 4, 4, 1), 4);
    }

    /// A width the contracted extent does not divide would leave a masked tail.
    #[test]
    #[should_panic(expected = "served in whole lines")]
    fn a_width_that_misdivides_the_contracted_axis_is_refused() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        contracted_per_step(&lhs, &rhs, &acc, 3, 3, 1);
    }

    /// A lined rhs has nothing to fold against when the lhs serves one value a step.
    #[test]
    #[should_panic(expected = "line the lhs along")]
    fn a_folded_step_needs_both_operands_lined() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        contracted_per_step(&lhs, &rhs, &acc, 1, 4, 1);
    }

    /// The block's lanes mean one axis, and a lined accumulator has already claimed them.
    #[test]
    #[should_panic(expected = "cannot also be served")]
    fn a_folded_step_needs_a_scalar_accumulator() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        contracted_per_step(&lhs, &rhs, &acc, 4, 4, 2);
    }

    /// The rhs and the accumulator share their line, so they share its width.
    #[test]
    #[should_panic(expected = "served at one width")]
    fn an_rhs_lined_along_the_accumulator_shares_its_width() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[K, N]);
        contracted_per_step(&lhs, &rhs, &acc, 4, 1, 2);
    }
}
