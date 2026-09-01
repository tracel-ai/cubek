//! The contraction nest's entry point: settle how many contracted values a step consumes, then
//! route to the 2-D or the N-D nest.

use cubecl::prelude::*;

use super::direct;
use super::gather;
use super::scale::{check_scales_omit_rather_than_divide, scale_side};
use super::shape::ContractShape;
use crate::*;

/// Run the register instruction over each batch matrix, reading operands through the
/// quant-transparent [`matrix_packed`](Tile::matrix_packed). Each operand resolves its own
/// [`Packing`], so neither side constrains the other's.
///
/// The 2-D nest reads each operand as a batch matrix, which describes it only when one axis is
/// contracted *and* a logical coordinate is a physical one. Either condition failing takes the
/// N-D nest, so a stencil contracting a single axis is a gather just as much as a two-axis
/// reduce is.
#[cube]
pub(crate) fn memory<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let lhs_procedural = lhs.is_procedural();
    let rhs_procedural = rhs.is_procedural();
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    let contracted_per_step = comptime!(step_contracted_per_step(
        &lhs.space, &rhs.space, &space, lw, rw, aw
    ));
    // Whether a 2-D reading describes the operands is the operands' own answer, not an axis count:
    // several contracted axes still form one `k` edge when the operand carries them as one run,
    // which is what a partitioned axis is.
    let shape = comptime!(ContractShape::new(
        &lhs.space,
        &rhs.space,
        space.clone(),
        contracted_per_step,
        lw,
        rw,
        aw
    ));
    let flat = comptime!(shape.matrix_axes(&lhs.space, &rhs.space).is_some());
    let nd = comptime!(
        !flat
            || lhs_gathered
            || rhs_gathered
            || lhs_procedural
            || rhs_procedural
            || (contracted_per_step == 1 && rw != aw)
    );

    if nd {
        gather::contract::<E, EL, ER>(acc, lhs, rhs, space, contracted_per_step, config, semiring);
    } else {
        direct::contract::<E, EL, ER>(acc, lhs, rhs, space, contracted_per_step, config, semiring);
    }
}

/// [`memory`] with one operand scaled by a real operand: `acc += (lhs ⊗ scale) · rhs`, or its
/// rhs twin, whichever [`scale_side`] reads off the scales' axes.
///
/// The 2-D nest only, deliberately. The N-D nest reads its operands through compacted gather
/// windows, where a step has no single scalar `k` to address a scale with; that is a second
/// design question, not a second copy of this one, and a routine reaching it gets told so here
/// rather than getting a wrong answer.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn memory_scaled<E: Numeric, EL: Numeric, ER: Numeric, ES: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Sequence<Tile<ES>>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let lhs_procedural = lhs.is_procedural();
    let rhs_procedural = rhs.is_procedural();
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    let contracted_per_step = comptime!(step_contracted_per_step(
        &lhs.space, &rhs.space, &space, lw, rw, aw
    ));
    // Same question the plain contraction asks: whether a 2-D reading describes the operands, not
    // how many axes they contract over.
    let shape = comptime!(ContractShape::new(
        &lhs.space,
        &rhs.space,
        space.clone(),
        contracted_per_step,
        lw,
        rw,
        aw
    ));
    let flat = comptime!(shape.matrix_axes(&lhs.space, &rhs.space).is_some());
    comptime!(assert!(
        flat && !lhs_gathered
            && !rhs_gathered
            && !lhs_procedural
            && !rhs_procedural
            && !(contracted_per_step == 1 && rw != aw),
        "mm_scaled: the scaled contraction reads each operand as one matrix. This one needs the \
         N-D nest, which addresses every operand at the cell instead"
    ));
    // Every query about "the scales" is about the level nearest the values: the coarser ones cover
    // a tile of its tiles, so they neither pick the side nor set the granularity.
    let inner = scales.index(0);
    let side = comptime!(scale_side(&inner.space, &space, shape.acc_axes));
    let scales_projection = inner.projection();
    comptime!(check_scales_omit_rather_than_divide(&scales_projection));
    direct::contract_scaled::<E, EL, ER, ES>(
        acc,
        lhs,
        rhs,
        scales,
        space,
        contracted_per_step,
        side,
        config,
        semiring,
    );
}

/// How many contracted values one step consumes, reconciled across both operands and the
/// accumulator.
///
/// Asked per operand ([`Space::contracted_per_step`]) because the answer differs per operand: an lhs lined
/// along the contracted axis folds, an rhs lined along the accumulator's innermost axis holds
/// cells that must stay apart. Both must serve the same count, and the block's lanes mean one
/// axis, so a folded step needs a scalar-contracted_per_step accumulator.
fn step_contracted_per_step(
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
        assert_eq!(step_contracted_per_step(&lhs, &rhs, &acc, 4, 2, 2), 1);
    }

    /// Both operands lined along the contracted axis: the lanes are partials of one cell.
    #[test]
    fn both_operands_lined_along_the_contracted_axis_serve_a_line() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        assert_eq!(step_contracted_per_step(&lhs, &rhs, &acc, 4, 4, 1), 4);
    }

    /// A width the contracted extent does not divide would leave a masked tail.
    #[test]
    #[should_panic(expected = "served in whole lines")]
    fn a_width_that_misdivides_the_contracted_axis_is_refused() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        step_contracted_per_step(&lhs, &rhs, &acc, 3, 3, 1);
    }

    /// A lined rhs has nothing to fold against when the lhs serves one value a step.
    #[test]
    #[should_panic(expected = "line the lhs along")]
    fn a_folded_step_needs_both_operands_lined() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        step_contracted_per_step(&lhs, &rhs, &acc, 1, 4, 1);
    }

    /// The block's lanes mean one axis, and a lined accumulator has already claimed them.
    #[test]
    #[should_panic(expected = "cannot also be served")]
    fn a_folded_step_needs_a_scalar_accumulator() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        step_contracted_per_step(&lhs, &rhs, &acc, 4, 4, 2);
    }

    /// The rhs and the accumulator share their line, so they share its width.
    #[test]
    #[should_panic(expected = "served at one width")]
    fn an_rhs_lined_along_the_accumulator_shares_its_width() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[K, N]);
        step_contracted_per_step(&lhs, &rhs, &acc, 4, 1, 2);
    }
}
