//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use super::scale::Side;
use super::shape::ContractShape;
use crate::instruction::registers::block;
use crate::*;

/// The contraction nest for a single contracted axis: over each batch matrix, the `mr × nr` block
/// of accumulators lives in registers (load once, `kc / contracted_per_step` steps, store once).
///
/// Each factor arrives as its values and the levels of scales that multiply them, innermost
/// first. A factor carrying none reads as its values alone, so this is the one nest whatever is
/// quantized: the scales are looked up at every line's own coordinates and the block below runs
/// the same contraction either way.
///
/// The 2-D form its reads assume: `mat` indexes a batch matrix, `(row, k)` and `(k, col)` (or
/// `(col, k)` at a folded step) address the operands. [`memory`](super::memory) routes anything
/// else to the N-D nest, so the conditions below are re-asserted rather than re-decided.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<E: Numeric, EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Scaled<EL, LS>,
    rhs: &Scaled<ER, RS>,
    #[comptime] space: Space,
    #[comptime] contracted_per_step: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_values = lhs.values();
    let rhs_values = rhs.values();
    let lhs_gathered = lhs_values.gathered();
    let rhs_gathered = rhs_values.gathered();
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered,
        "contract: a gathered operand has no 2-D matrix view; it needs the N-D nest"
    ));

    let lw = lhs_values.vector_size();
    let rw = rhs_values.vector_size();
    let aw = comptime!(acc.store.vector_size);
    comptime!(assert!(
        rw == aw || contracted_per_step > 1,
        "contract direct: a padded rhs staged wider than its {aw}-wide sink must use the N-D nest"
    ));
    let shape = comptime!(ContractShape::new(
        &lhs_values.space,
        &rhs_values.space,
        space,
        contracted_per_step,
        lw,
        rw,
        aw,
    ));
    comptime!(assert!(
        shape
            .matrix_axes(&lhs_values.space, &rhs_values.space)
            .is_some(),
        "contract: the 2-D nest reads each operand as one matrix, and no grouping of these axes \
         gives one; the N-D nest reads them a cell at a time"
    ));

    // The block's lines are the rhs's: `contracted_per_step`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(contracted_per_step > 1) {
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        nest::<E, EL, W, LS, ER, W, RS, A>(acc, lhs, rhs, shape, config, semiring);
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest::<E, EL, W, LS, ER, A, RS, A>(acc, lhs, rhs, shape, config, semiring);
    }
}

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and so the block's, `A` the
/// accumulator's.
#[cube]
#[allow(clippy::too_many_arguments)]
fn nest<
    E: Numeric,
    EL: Numeric,
    L: Size,
    LS: Numeric,
    ER: Numeric,
    V: Size,
    RS: Numeric,
    A: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Scaled<EL, LS>,
    rhs: &Scaled<ER, RS>,
    #[comptime] shape: ContractShape,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_values = lhs.values();
    let rhs_values = rhs.values();
    let mr = comptime!(shape.mr);
    let nr = comptime!(shape.nr);
    let cols = comptime!(shape.cols);
    let kc = comptime!(shape.kc);
    let contracted_per_step = comptime!(shape.contracted_per_step);
    let lw = comptime!(shape.lw);
    let aw = comptime!(shape.aw);
    let matrices = comptime!(shape.matrices());

    let lhs_axes = comptime!(shape.lhs_axes(&lhs_values.space));
    let rhs_axes = comptime!(shape.rhs_axes(&rhs_values.space));

    // Only the bound proof below needs the lhs's line count; the walk itself splits `kc`.
    let lhs_k_lines = comptime!(kc.div_ceil(lw));
    let lane_fanout = comptime!(config.lane_fanout);

    for mat in 0..matrices {
        let lhs_mat = lhs_values.matrix_packed::<L>(lhs_axes, mat);
        let rhs_mat = rhs_values.matrix_packed::<V>(rhs_axes, mat);
        // Each factor's scales, looked up at its lines' own coordinates.
        let lhs_scales = lhs.lookup(
            lhs_axes,
            mat,
            comptime!(Side::Lhs),
            comptime!(shape.space.clone()),
            comptime!(shape.acc_axes),
        );
        let rhs_scales = rhs.lookup(
            rhs_axes,
            mat,
            comptime!(Side::Rhs),
            comptime!(shape.space.clone()),
            comptime!(shape.acc_axes),
        );
        // The contraction's own algebra: its products accumulate under the semiring's add.
        let mut acc_view = acc.matrix_accumulate::<A>(
            mat,
            comptime!(shape.acc_axes),
            comptime!(shape.space.clone()),
            comptime!(semiring.add()),
        );

        // A checked edge normally rolls every local array access. When enabled, split the leaf
        // into two comptime-specialized bodies: interior instances prove their complete operand
        // and accumulator blocks in bounds once, then keep `c` and `b` in registers; only the
        // actual edge instance pays masked reads/writes and runtime local-array indexing.
        let lhs_check = comptime!(lhs_mat.check);
        let rhs_check = comptime!(rhs_mat.check);
        let acc_check = acc_view.check();
        let eligible = comptime!(shape.scalars() <= config.budget);
        let split_edge =
            comptime!(eligible && config.split_edge && (lhs_check || rhs_check || acc_check));
        let in_bounds = if comptime!(split_edge) {
            let origin = (0u32.runtime(), 0u32.runtime());
            let lhs_extent = (
                comptime!(mr as u32).runtime(),
                comptime!(lhs_k_lines as u32).runtime(),
            );
            let rhs_extent = if comptime!(contracted_per_step > 1) {
                (
                    comptime!(nr as u32).runtime(),
                    comptime!((kc / contracted_per_step) as u32).runtime(),
                )
            } else {
                (
                    comptime!(kc as u32).runtime(),
                    comptime!(nr as u32).runtime(),
                )
            };
            let acc_extent = (
                comptime!(mr as u32).runtime(),
                comptime!(nr as u32).runtime(),
            );
            lhs_mat.block_in_bounds(origin, lhs_extent)
                && rhs_mat.block_in_bounds(origin, rhs_extent)
                && acc_view.block_in_bounds(origin, acc_extent)
        } else {
            false.runtime()
        };

        if comptime!(split_edge) {
            if in_bounds {
                body::<E, EL, L, LS, ER, V, RS, A>(
                    &mut acc_view,
                    &lhs_mat,
                    &lhs_scales,
                    &rhs_mat,
                    &rhs_scales,
                    lw,
                    contracted_per_step,
                    aw,
                    mr,
                    nr,
                    cols,
                    kc,
                    true,
                    lane_fanout,
                    semiring,
                );
            } else {
                body::<E, EL, L, LS, ER, V, RS, A>(
                    &mut acc_view,
                    &lhs_mat,
                    &lhs_scales,
                    &rhs_mat,
                    &rhs_scales,
                    lw,
                    contracted_per_step,
                    aw,
                    mr,
                    nr,
                    cols,
                    kc,
                    false,
                    lane_fanout,
                    semiring,
                );
            }
        } else {
            let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);
            body::<E, EL, L, LS, ER, V, RS, A>(
                &mut acc_view,
                &lhs_mat,
                &lhs_scales,
                &rhs_mat,
                &rhs_scales,
                lw,
                contracted_per_step,
                aw,
                mr,
                nr,
                cols,
                kc,
                unroll,
                lane_fanout,
                semiring,
            );
        }
    }
}

/// The complete nest body, specialized at trace time for either register-resident local arrays
/// (`unroll = true`) or the checked edge fallback (`unroll = false`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn body<
    E: Numeric,
    EL: Numeric,
    L: Size,
    LS: Numeric,
    ER: Numeric,
    V: Size,
    RS: Numeric,
    A: Size,
>(
    acc: &mut AccumulateView<'_, E, A>,
    lhs: &MatrixView<'_, Vector<EL, L>>,
    lhs_scales: &ScaleLookup<LS>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    rhs_scales: &ScaleLookup<RS>,
    #[comptime] lw: usize,
    #[comptime] contracted_per_step: usize,
    #[comptime] aw: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
    #[comptime] semiring: Semiring,
) {
    let mut c = block::seed::<E, V, A>(acc, contracted_per_step, 1usize, aw, mr, nr, cols, unroll);
    block::contract::<E, EL, L, LS, ER, V, RS>(
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        &mut c,
        lw,
        contracted_per_step,
        mr,
        nr,
        kc,
        unroll,
        lane_fanout,
        semiring,
    );
    block::commit::<E, V, A>(
        acc,
        c,
        contracted_per_step,
        1usize,
        aw,
        mr,
        nr,
        cols,
        unroll,
    );
}
