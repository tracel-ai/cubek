//! The contraction nest's entry point: settle how many contracted values a step consumes, then
//! route to the 2-D or the N-D nest.

use cubecl::prelude::*;

use super::direct;
use super::gather;
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
    let served = comptime!(step_served(&lhs.space, &rhs.space, &space, lw, rw, aw));
    // Whether a 2-D reading describes the operands is the operands' own answer, not an axis count:
    // several contracted axes still form one `k` edge when the operand carries them as one run,
    // which is what a partitioned axis is.
    let flat = comptime!(
        ContractShape::new(&lhs.space, &rhs.space, space.clone(), served, lw, rw, aw)
            .matrix_axes(&lhs.space, &rhs.space)
            .is_some()
    );
    let nd = comptime!(
        !flat
            || lhs_gathered
            || rhs_gathered
            || lhs_procedural
            || rhs_procedural
            || (served == 1 && rw != aw)
    );

    if nd {
        gather::contract::<E, EL, ER>(acc, lhs, rhs, space, served, config, semiring);
    } else {
        direct::contract::<E, EL, ER>(acc, lhs, rhs, space, served, config, semiring);
    }
}

/// Which factor of the term a scales operand multiplies. Read off the axes it spans, never
/// stated: a scale over the accumulator's column axis is a fact about the rhs's columns and
/// nothing else could fold it in; anything else scales the lhs.
///
/// One verb, then, not two. `(a ⊗ s) · b` and `a · (b ⊗ s)` are the same sum of terms — the scale
/// is one more factor of each — and which operand it rides is only *where* it folds in cheapest:
/// once per `(row, k)` beside the lhs, or once per `(col, k)` beside the rhs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleSide {
    /// The scale spans the accumulator's rows (or the contracted axis alone): folded into the
    /// lhs value before it forms its products.
    Lhs,
    /// The scale spans the accumulator's columns: folded into each rhs line.
    Rhs,
}

/// The side a scales operand multiplies on, from the axes it spans against the accumulator's.
///
/// A scale over neither matrix axis (per-tensor, or one value per block of `k`) is the same
/// number wherever it folds, so it takes the lhs side.
pub(crate) fn scale_side(scales: &Space, output: &Space) -> ScaleSide {
    let rank = output.rank();
    let (rows, cols) = (output.axis_at(rank - 2), output.axis_at(rank - 1));
    let spans = |axis| scales.axes().any(|a| a == axis);
    assert!(
        !(spans(rows) && spans(cols)),
        "mm_scaled: a scales operand over both {rows:?} and {cols:?} is a scale of the output, \
         not a factor of either operand's term"
    );
    match spans(cols) {
        true => ScaleSide::Rhs,
        false => ScaleSide::Lhs,
    }
}

/// The block a scales operand resolves `axis` at: the divisor of the physical axis addressing it,
/// so `PhysicalAxisMap::of(K).over(8)` answers `8` and a plain axis answers `1`. An axis the
/// operand does not span answers [`usize::MAX`]: one scale covers every value of it, so no line
/// along it can straddle anything.
pub(crate) fn scale_block(projection: &Projection, axis: Axis) -> usize {
    (0..projection.physical_rank())
        .find(|&pa| projection.scale(pa, axis) == 1)
        .map(|pa| projection.divisor(pa).bound())
        .unwrap_or(usize::MAX)
}

/// Refuse a served line that straddles two scales: one line is one read and takes one scale, so
/// the block along whichever axis the line runs must cover whole lines. The line runs along the
/// *innermost* contracted axis, which is the one a partitioned contraction leaves at the leaf.
///
/// Which axis that is depends on the step: past one served value both operands line along the
/// contracted axis, and at one the rhs lines along the accumulator's columns instead. The lhs at
/// one served value takes a scalar `k` per step and lines along nothing.
pub(crate) fn check_lines_hold_one_scale(
    scales: &Projection,
    k: Axis,
    cols: Axis,
    served: usize,
    aw: usize,
    side: ScaleSide,
) {
    let (axis, width) = match (served > 1, side) {
        (true, _) => (k, served),
        (false, ScaleSide::Rhs) => (cols, aw),
        (false, ScaleSide::Lhs) => return,
    };
    let block = scale_block(scales, axis);
    assert!(
        block == usize::MAX || block.is_multiple_of(width),
        "mm_scaled: a step reads {width} values of {axis:?} as one line, so its {block}-value \
         scale blocks must cover whole lines; state a block the line divides, or a cut that \
         serves narrower lines"
    );
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
    scales: &Tile<ES>,
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
    let served = comptime!(step_served(&lhs.space, &rhs.space, &space, lw, rw, aw));
    // Same question the plain contraction asks: whether a 2-D reading describes the operands, not
    // how many axes they contract over.
    let flat = comptime!(
        ContractShape::new(&lhs.space, &rhs.space, space.clone(), served, lw, rw, aw)
            .matrix_axes(&lhs.space, &rhs.space)
            .is_some()
    );
    comptime!(assert!(
        flat && !lhs_gathered
            && !rhs_gathered
            && !lhs_procedural
            && !rhs_procedural
            && !(served == 1 && rw != aw),
        "mm_scaled: the scaled contraction reads each operand as one matrix. This one needs the \
         N-D nest, which addresses every operand at the cell instead"
    ));
    let side = comptime!(scale_side(&scales.space, &space));
    let scales_projection = scales.projection();
    comptime!(check_lines_hold_one_scale(
        &scales_projection,
        *Space::contracted(&[&lhs.space, &rhs.space], &space)
            .last()
            .unwrap(),
        space.axis_at(space.rank() - 1),
        served,
        aw,
        side,
    ));
    direct::contract_scaled::<E, EL, ER, ES>(
        acc, lhs, rhs, scales, space, served, side, config, semiring,
    );
}

/// How many contracted values one step consumes, reconciled across both operands and the
/// accumulator.
///
/// Asked per operand ([`Space::served`]) because the answer differs per operand: an lhs lined
/// along the contracted axis folds, an rhs lined along the accumulator's innermost axis holds
/// cells that must stay apart. Both must serve the same count, and the block's lanes mean one
/// axis, so a folded step needs a scalar-served accumulator.
fn step_served(lhs: &Space, rhs: &Space, acc: &Space, lw: usize, rw: usize, aw: usize) -> usize {
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
    let served = rhs.served(&contracted, rw);
    assert!(
        served > 1,
        "contract: the rhs lines along the contracted axis {k:?}, which is served in whole lines; \
         its width {rw} must exceed 1 and divide the axis's extent {}",
        rhs.extent(k)
    );
    assert_eq!(
        lhs.served(&contracted, lw),
        served,
        "contract: the rhs serves {served} contracted values a step; line the lhs along {k:?} at \
         the same width (it is {lw} wide)"
    );
    assert_eq!(
        aw, 1,
        "contract: a step serving {served} contracted values holds partials of one cell in the \
         block's lanes, so the accumulator cannot also be served in {aw}-wide lines"
    );
    served
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
        assert_eq!(step_served(&lhs, &rhs, &acc, 4, 2, 2), 1);
    }

    /// Both operands lined along the contracted axis: the lanes are partials of one cell.
    #[test]
    fn both_operands_lined_along_the_contracted_axis_serve_a_line() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        assert_eq!(step_served(&lhs, &rhs, &acc, 4, 4, 1), 4);
    }

    /// A width the contracted extent does not divide would leave a masked tail.
    #[test]
    #[should_panic(expected = "served in whole lines")]
    fn a_width_that_misdivides_the_contracted_axis_is_refused() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        step_served(&lhs, &rhs, &acc, 3, 3, 1);
    }

    /// A lined rhs has nothing to fold against when the lhs serves one value a step.
    #[test]
    #[should_panic(expected = "line the lhs along")]
    fn a_folded_step_needs_both_operands_lined() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        step_served(&lhs, &rhs, &acc, 1, 4, 1);
    }

    /// The block's lanes mean one axis, and a lined accumulator has already claimed them.
    #[test]
    #[should_panic(expected = "cannot also be served")]
    fn a_folded_step_needs_a_scalar_accumulator() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        step_served(&lhs, &rhs, &acc, 4, 4, 2);
    }

    /// The rhs and the accumulator share their line, so they share its width.
    #[test]
    #[should_panic(expected = "served at one width")]
    fn an_rhs_lined_along_the_accumulator_shares_its_width() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[K, N]);
        step_served(&lhs, &rhs, &acc, 4, 1, 2);
    }
}
