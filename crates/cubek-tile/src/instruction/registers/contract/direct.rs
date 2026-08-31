//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use super::shape::ContractShape;
use crate::instruction::registers::block;
use crate::instruction::registers::contract::ScaleSide;
use crate::instruction::registers::lines::{Lines, ScaledLines};
use crate::*;

/// The contraction nest for a single contracted axis: over each batch matrix, the `mr × nr` block
/// of accumulators lives in registers (load once, `kc / contracted_per_step` steps, store once).
///
/// The 2-D form its reads assume: `mat` indexes a batch matrix, `(row, k)` and `(k, col)` (or
/// `(col, k)` at a folded step) address the operands. [`memory`](super::memory) routes anything
/// else to the N-D nest, so the conditions below are re-asserted rather than re-decided.
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

    // The block's lines are the rhs's: `contracted_per_step`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(contracted_per_step > 1) {
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, shape, config, semiring);
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest::<E, EL, W, ER, A, A>(acc, lhs, rhs, shape, config, semiring);
    }
}

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and so the block's, `A` the
/// accumulator's.
#[cube]
fn nest<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
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
        if comptime!(split_edge) {
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
            let in_bounds = lhs_mat.block_in_bounds(origin, lhs_extent)
                && rhs_mat.block_in_bounds(origin, rhs_extent)
                && acc_view.block_in_bounds(origin, acc_extent);
            if in_bounds {
                body::<E, EL, L, ER, V, A, MatrixView<Vector<EL, L>>, MatrixView<Vector<ER, V>>>(
                    &mut acc_view,
                    &lhs_mat,
                    &rhs_mat,
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
                body::<E, EL, L, ER, V, A, MatrixView<Vector<EL, L>>, MatrixView<Vector<ER, V>>>(
                    &mut acc_view,
                    &lhs_mat,
                    &rhs_mat,
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
            body::<E, EL, L, ER, V, A, MatrixView<Vector<EL, L>>, MatrixView<Vector<ER, V>>>(
                &mut acc_view,
                &lhs_mat,
                &rhs_mat,
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
    Lhs: Lines<EL, L>,
    Rhs: Lines<ER, V>,
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

/// [`contract`] with one operand scaled: `c += (lhs ⊗ scale) · rhs` or `c += lhs · (rhs ⊗ scale)`,
/// the scale a real operand read through its own view and [`ScaleSide`] saying which factor it
/// meets. Same nest, same block: the scale folds in under the operand's view, so the block
/// below runs the plain contraction.
///
/// The scales are read where the values are, never staged: one value per block is already
/// cache-served, and staging one would materialize the expansion reading it in place avoids.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract_scaled<E: Numeric, EL: Numeric, ER: Numeric, ES: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Tile<ES>,
    #[comptime] space: Space,
    #[comptime] contracted_per_step: usize,
    #[comptime] side: ScaleSide,
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
    let aw = comptime!(acc.store.vector_size);
    let rw = rhs.vector_size();
    let sw = scales.vector_size();
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

    let size!(S) = sw;
    if comptime!(contracted_per_step > 1) {
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        nest_scaled::<E, EL, W, ER, W, A, ES, S>(
            acc, lhs, rhs, scales, shape, side, sw, config, semiring,
        );
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest_scaled::<E, EL, W, ER, A, A, ES, S>(
            acc, lhs, rhs, scales, shape, side, sw, config, semiring,
        );
    }
}

/// [`nest`] with the scales view built beside the operands'.
#[cube]
#[allow(clippy::too_many_arguments)]
fn nest_scaled<
    E: Numeric,
    EL: Numeric,
    L: Size,
    ER: Numeric,
    V: Size,
    A: Size,
    ES: Numeric,
    S: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Tile<ES>,
    #[comptime] shape: ContractShape,
    #[comptime] side: ScaleSide,
    #[comptime] sw: usize,
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
    let space = comptime!(shape.space.clone());
    let lane_fanout = comptime!(config.lane_fanout);

    let lhs_axes = comptime!(shape.lhs_axes(&lhs.space));
    let rhs_axes = comptime!(shape.rhs_axes(&rhs.space));
    // The scales share one edge with the values: the contraction where a folded step or the lhs
    // carries them, the accumulator's columns otherwise. `value_width` is the width that edge is
    // served at, and each axis is paired with its extent because a contracted one is not the
    // accumulator's to size.
    let scale_cols = comptime!(scales.space.extent_at(scales.space.rank() - 1));
    let (scales_axes, edge, value_width) = comptime!(match (side, contracted_per_step > 1) {
        (ScaleSide::Lhs, _) => (
            MatrixAxes::of(&scales.space, mr, scale_cols),
            shape.reduce_edge(),
            lw
        ),
        (ScaleSide::Rhs, false) => (
            MatrixAxes::of(&scales.space, kc, scale_cols),
            shape.column_edge(),
            aw
        ),
        (ScaleSide::Rhs, true) => (
            MatrixAxes::of(&scales.space, cols, scale_cols),
            shape.reduce_edge(),
            contracted_per_step
        ),
    });
    // How many value lines share one scale, read off the axes rather than divided out of the
    // extents: the scale is constant along the edge axes it does not distinguish, so one read of
    // it serves every position of them.
    let operands = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let invariant = scales.invariant_over(operands);
    let lines_per_scale = comptime!(
        edge.iter()
            .filter(|(axis, _)| invariant.contains(axis))
            .map(|(_, extent)| *extent)
            .product::<usize>()
            / value_width
    );
    // A scale line wider than one scale needs each value line's ordinal along the shared edge as a
    // constant. The rhs's columns are walked under one at a step; the lhs's are the contraction,
    comptime!(assert!(
        sw == 1 || contracted_per_step == 1,
        "mm_scaled: {sw} scales are served as one line, which needs each value line's ordinal \
         along the shared edge as a constant. A step folding {contracted_per_step} contracted \
         values walks no such edge; bind the scales scalar here"
    ));
    let eligible = comptime!(mr * nr * contracted_per_step * aw <= config.budget);

    for mat in 0..matrices {
        let mut acc_view = acc.matrix_accumulate::<A>(
            mat,
            comptime!(shape.acc_axes),
            comptime!(space.clone()),
            comptime!(semiring.add()),
        );
        let acc_check = acc_view.check();

        // The scale folds into the operand that carries it, so the block below contracts one
        // scaled line source against one plain one and runs the same body either way. Which
        // operand that is decides two types, so it decides two calls.
        match comptime!(side) {
            ScaleSide::Lhs => {
                let values = lhs.matrix_packed::<L>(lhs_axes, mat);
                let rhs_mat = rhs.matrix_packed::<V>(rhs_axes, mat);
                let unroll = comptime!(eligible && !values.check && !rhs_mat.check && !acc_check);
                let lhs_mat = ScaledLines::<EL, L, ES, S>::new(
                    values,
                    scales.matrix_packed::<S>(scales_axes, mat),
                    lines_per_scale,
                    sw,
                );
                body::<E, EL, L, ER, V, A, ScaledLines<EL, L, ES, S>, MatrixView<Vector<ER, V>>>(
                    &mut acc_view,
                    &lhs_mat,
                    &rhs_mat,
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
            ScaleSide::Rhs => {
                let lhs_mat = lhs.matrix_packed::<L>(lhs_axes, mat);
                let values = rhs.matrix_packed::<V>(rhs_axes, mat);
                let unroll = comptime!(eligible && !lhs_mat.check && !values.check && !acc_check);
                let rhs_mat = ScaledLines::<ER, V, ES, S>::new(
                    values,
                    scales.matrix_packed::<S>(scales_axes, mat),
                    lines_per_scale,
                    sw,
                );
                body::<E, EL, L, ER, V, A, MatrixView<Vector<EL, L>>, ScaledLines<ER, V, ES, S>>(
                    &mut acc_view,
                    &lhs_mat,
                    &rhs_mat,
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
}
