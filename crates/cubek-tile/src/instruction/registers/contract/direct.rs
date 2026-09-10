//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use super::factor::{level_of, scale_width, scales_of, spread};
use super::scale::{ContractEdges, EdgeOrdinal, Side};
use super::shape::ContractShape;
use crate::instruction::registers::block;
use crate::instruction::registers::lines::{CombinedScales, Lines, ScaledLines};
use crate::*;

/// The contraction nest for a single contracted axis: over each batch matrix, the `mr × nr` block
/// of accumulators lives in registers (load once, `kc / contracted_per_step` steps, store once).
///
/// Each factor arrives as its values and the levels of scales that multiply them, innermost
/// first. A factor carrying none reads as its values alone, so this is the one nest whatever is
/// quantized: the scales fold in under the factor's own line source and the block below runs the
/// same contraction either way.
///
/// The 2-D form its reads assume: `mat` indexes a batch matrix, `(row, k)` and `(k, col)` (or
/// `(col, k)` at a folded step) address the operands. [`memory`](super::memory) routes anything
/// else to the N-D nest, so the conditions below are re-asserted rather than re-decided.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<E: Numeric, EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    lhs_levels: &Sequence<Tile<LS>>,
    rhs: &Tile<ER>,
    rhs_levels: &Sequence<Tile<RS>>,
    #[comptime] space: Space,
    #[comptime] contracted_per_step: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered,
        "contract: a gathered operand has no 2-D matrix view; it needs the N-D nest"
    ));

    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    comptime!(assert!(
        rw == aw || contracted_per_step > 1,
        "contract direct: a padded rhs staged wider than its {aw}-wide sink must use the N-D nest"
    ));
    let shape = comptime!(ContractShape::new(
        &lhs.space,
        &rhs.space,
        space,
        contracted_per_step,
        lw,
        rw,
        aw,
    ));
    comptime!(assert!(
        shape.matrix_axes(&lhs.space, &rhs.space).is_some(),
        "contract: the 2-D nest reads each operand as one matrix, and no grouping of these axes \
         gives one; the N-D nest reads them a cell at a time"
    ));

    // Each factor's scales are read at their own width; one that carries none reads nothing.
    let lsw = scale_width(lhs_levels);
    let rsw = scale_width(rhs_levels);
    let size!(LSW) = lsw;
    let size!(RSW) = rsw;

    // The block's lines are the rhs's: `contracted_per_step`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(contracted_per_step > 1) {
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        nest::<E, EL, W, LS, LSW, ER, W, RS, RSW, A>(
            acc, lhs, lhs_levels, rhs, rhs_levels, shape, config, semiring,
        );
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest::<E, EL, W, LS, LSW, ER, A, RS, RSW, A>(
            acc, lhs, lhs_levels, rhs, rhs_levels, shape, config, semiring,
        );
    }
}

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and so the block's, `A` the
/// accumulator's, `LSW` and `RSW` the widths each factor's scales are read at.
#[cube]
#[allow(clippy::too_many_arguments)]
fn nest<
    E: Numeric,
    EL: Numeric,
    L: Size,
    LS: Numeric,
    LSW: Size,
    ER: Numeric,
    V: Size,
    RS: Numeric,
    RSW: Size,
    A: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    lhs_levels: &Sequence<Tile<LS>>,
    rhs: &Tile<ER>,
    rhs_levels: &Sequence<Tile<RS>>,
    #[comptime] shape: ContractShape,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(shape.mr);
    let nr = comptime!(shape.nr);
    let cols = comptime!(shape.cols);
    let kc = comptime!(shape.kc);
    let contracted_per_step = comptime!(shape.contracted_per_step);
    let lw = comptime!(shape.lw);
    let aw = comptime!(shape.aw);
    let matrices = comptime!(shape.matrices());

    let lhs_axes = comptime!(shape.lhs_axes(&lhs.space));
    let rhs_axes = comptime!(shape.rhs_axes(&rhs.space));

    // This nest's own edges, which is all a scale level needs of it; what a level *is* against
    // the values it covers is `ScaleLevel`'s to read, here and in the promoted block alike.
    let operands = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let edges = comptime!(ContractEdges {
        mr,
        kc,
        cols,
        reduce: shape.reduce_edge(),
        columns: shape.column_edge(),
        lw,
        aw,
        contracted_per_step,
        // A step folding several contracted values takes them from one line at a runtime index;
        // an unfolded one walks its lines under a constant ordinal.
        ordinal: match contracted_per_step {
            1 => EdgeOrdinal::Constant,
            folded => EdgeOrdinal::Runtime(format!(
                "a step folding {folded} contracted values walks no such edge"
            )),
        },
    });
    let lhs_level = level_of(
        lhs_levels,
        comptime!(operands.clone()),
        comptime!(shape.space.clone()),
        comptime!(shape.acc_axes),
        comptime!(edges.clone()),
        comptime!(Side::Lhs),
    );
    let rhs_level = level_of(
        rhs_levels,
        comptime!(operands.clone()),
        comptime!(shape.space.clone()),
        comptime!(shape.acc_axes),
        comptime!(edges),
        comptime!(Side::Rhs),
    );
    let lhs_spread = comptime!(spread(lhs_level));
    let rhs_spread = comptime!(spread(rhs_level));

    // Only the bound proof below needs the lhs's line count; the walk itself splits `kc`.
    let lhs_k_lines = comptime!(kc.div_ceil(lw));
    let lane_fanout = comptime!(config.lane_fanout);

    for mat in 0..matrices {
        let lhs_mat = lhs.matrix_packed::<L>(lhs_axes, mat);
        let rhs_mat = rhs.matrix_packed::<V>(rhs_axes, mat);
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

        // Each factor as the block reads it: its values, times whatever scales it carries.
        let lhs_f = ScaledLines::<MatrixView<Vector<EL, L>>, CombinedScales<LS, LSW>>::new(
            lhs_mat,
            scales_of::<LS, LSW>(lhs_levels, comptime!(lhs_level), mat),
            comptime!(lhs_spread.0),
            comptime!(lhs_spread.1),
        );
        let rhs_f = ScaledLines::<MatrixView<Vector<ER, V>>, CombinedScales<RS, RSW>>::new(
            rhs_mat,
            scales_of::<RS, RSW>(rhs_levels, comptime!(rhs_level), mat),
            comptime!(rhs_spread.0),
            comptime!(rhs_spread.1),
        );

        if comptime!(split_edge) {
            if in_bounds {
                body::<
                    E,
                    EL,
                    L,
                    ER,
                    V,
                    A,
                    ScaledLines<MatrixView<Vector<EL, L>>, CombinedScales<LS, LSW>>,
                    ScaledLines<MatrixView<Vector<ER, V>>, CombinedScales<RS, RSW>>,
                >(
                    &mut acc_view,
                    &lhs_f,
                    &rhs_f,
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
                body::<
                    E,
                    EL,
                    L,
                    ER,
                    V,
                    A,
                    ScaledLines<MatrixView<Vector<EL, L>>, CombinedScales<LS, LSW>>,
                    ScaledLines<MatrixView<Vector<ER, V>>, CombinedScales<RS, RSW>>,
                >(
                    &mut acc_view,
                    &lhs_f,
                    &rhs_f,
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
            body::<
                E,
                EL,
                L,
                ER,
                V,
                A,
                ScaledLines<MatrixView<Vector<EL, L>>, CombinedScales<LS, LSW>>,
                ScaledLines<MatrixView<Vector<ER, V>>, CombinedScales<RS, RSW>>,
            >(
                &mut acc_view,
                &lhs_f,
                &rhs_f,
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
    ER: Numeric,
    V: Size,
    A: Size,
    Lhs: Lines<E = EL, V = L>,
    Rhs: Lines<E = ER, V = V>,
>(
    acc: &mut AccumulateView<'_, E, A>,
    lhs: &Lhs,
    rhs: &Rhs,
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
    block::contract::<E, EL, L, ER, V, Lhs, Rhs>(
        lhs,
        rhs,
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
