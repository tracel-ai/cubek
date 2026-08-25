//! The contraction nest's entry point: settle how many contracted values a step consumes, then
//! route to the 2-D or the N-D nest.

use cubecl::prelude::*;

use super::direct;
use super::gather;
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
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let lhs_procedural = lhs.is_procedural();
    let rhs_procedural = rhs.is_procedural();
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    let served = comptime!(step_served(&lhs.space, &rhs.space, &space, lw, rw, aw));
    let nd = comptime!(
        Space::contracted(&[&lhs.space, &rhs.space], &space).len() > 1
            || lhs_gathered
            || rhs_gathered
            || lhs_procedural
            || rhs_procedural
            || (served == 1 && rw != aw)
    );

    if nd {
        gather::contract::<E, EL, ER>(acc, lhs, rhs, space, served, config);
    } else {
        direct::contract::<E, EL, ER>(acc, lhs, rhs, space, served, config);
    }
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
