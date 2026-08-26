//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use super::shape::ContractShape;
use crate::instruction::registers::block;
use crate::instruction::registers::contract::ScaleSide;
use crate::*;

/// The contraction nest for a single contracted axis: over each batch matrix, the `mr × nr` block
/// of accumulators lives in registers (load once, `kc / served` steps, store once).
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
    #[comptime] served: usize,
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
        rw == aw || served > 1,
        "contract direct: a padded rhs staged wider than its {aw}-wide sink must use the N-D nest"
    ));
    let shape = comptime!(ContractShape::new(
        &lhs.space, &rhs.space, space, served, lw, rw, aw,
    ));
    comptime!(assert!(
        shape.matrix_axes(&lhs.space, &rhs.space).is_some(),
        "contract: the 2-D nest reads each operand as one matrix, and no grouping of these axes \
         gives one; the N-D nest reads them a cell at a time"
    ));

    // The block's lines are the rhs's: `served`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(served > 1) {
        let size!(W) = served;
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
    let served = comptime!(shape.served);
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
            let rhs_extent = if comptime!(served > 1) {
                (
                    comptime!(nr as u32).runtime(),
                    comptime!((kc / served) as u32).runtime(),
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
                body::<E, EL, L, ER, V, A>(
                    &mut acc_view,
                    &lhs_mat,
                    &rhs_mat,
                    lw,
                    served,
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
                body::<E, EL, L, ER, V, A>(
                    &mut acc_view,
                    &lhs_mat,
                    &rhs_mat,
                    lw,
                    served,
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
            body::<E, EL, L, ER, V, A>(
                &mut acc_view,
                &lhs_mat,
                &rhs_mat,
                lw,
                served,
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
fn body<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    #[comptime] lw: usize,
    #[comptime] served: usize,
    #[comptime] aw: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
    #[comptime] semiring: Semiring,
) {
    let mut c = block::seed::<E, V, A>(acc, served, 1usize, aw, mr, nr, cols, unroll);
    block::contract::<E, EL, L, ER, V>(
        lhs,
        rhs,
        &mut c,
        lw,
        served,
        mr,
        nr,
        kc,
        unroll,
        lane_fanout,
        semiring,
    );
    block::commit::<E, V, A>(acc, c, served, 1usize, aw, mr, nr, cols, unroll);
}

/// [`contract`] with one operand scaled: `c += (lhs ⊗ scale) · rhs` or `c += lhs · (rhs ⊗ scale)`,
/// the scale a real operand read through its own view and [`ScaleSide`] saying which factor it
/// meets. Same nest, same block: the scale folds in under the operand's view, so the block
/// below runs the plain contraction.
///
/// The scales are read where the values are, never staged: one value per block is already
/// cache-served, and a stage would materialize the expansion the coarse read exists to avoid.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract_scaled<E: Numeric, EL: Numeric, ER: Numeric, ES: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Tile<ES>,
    #[comptime] space: Space,
    #[comptime] served: usize,
    #[comptime] side: ScaleSide,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    comptime!(assert!(
        Space::contracted(&[&lhs.space, &rhs.space], &space).len() == 1,
        "contract: the 2-D nest contracts exactly one axis"
    ));
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
        sw == 1,
        "mm_scaled: the scales are read one value at a time, so their operand is scalar (it is \
         {sw} wide)"
    ));
    comptime!(assert!(
        rw == aw || served > 1,
        "contract direct: a padded rhs staged wider than its {aw}-wide sink must use the N-D nest"
    ));

    let size!(S) = 1usize;
    if comptime!(served > 1) {
        let size!(W) = served;
        let size!(A) = 1usize;
        nest_scaled::<E, EL, W, ER, W, A, ES, S>(
            acc, lhs, rhs, scales, space, served, lw, 1usize, side, config, semiring,
        );
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest_scaled::<E, EL, W, ER, A, A, ES, S>(
            acc, lhs, rhs, scales, space, served, lw, aw, side, config, semiring,
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
    #[comptime] space: Space,
    #[comptime] served: usize,
    #[comptime] lw: usize,
    #[comptime] aw: usize,
    #[comptime] side: ScaleSide,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let rank = comptime!(space.rank());
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let k = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space)[0]);
    let kc = comptime!(merged.extent(k));

    let cols = comptime!(space.extent_at(rank - 1));
    let (mr, nr) = comptime!((space.extent_at(rank - 2), cols / aw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());
    let lane_fanout = comptime!(config.lane_fanout);

    let lhs_axes = comptime!(MatrixAxes::of(&lhs.space, mr, kc));
    let rhs_axes = comptime!(match served > 1 {
        true => MatrixAxes::of(&rhs.space, cols, kc),
        false => MatrixAxes::of(&rhs.space, kc, cols),
    });
    let scales_axes = comptime!(MatrixAxes::trailing_pair(&scales.space));

    for mat in 0..matrices {
        // The scale folds in under the view, so what the block below contracts is two ordinary
        // matrices and the plain body runs them.
        let lhs_mat = match comptime!(side) {
            ScaleSide::Lhs => lhs.matrix_scaled::<L, ES, S>(lhs_axes, scales, scales_axes, mat),
            ScaleSide::Rhs => lhs.matrix_packed::<L>(lhs_axes, mat),
        };
        let rhs_mat = match comptime!(side) {
            ScaleSide::Lhs => rhs.matrix_packed::<V>(rhs_axes, mat),
            ScaleSide::Rhs => rhs.matrix_scaled::<V, ES, S>(rhs_axes, scales, scales_axes, mat),
        };
        let mut acc_view =
            acc.matrix_accumulate::<A>(mat, comptime!(space.clone()), comptime!(semiring.add()));

        let lhs_check = comptime!(lhs_mat.check);
        let rhs_check = comptime!(rhs_mat.check);
        let acc_check = acc_view.check();
        let eligible = comptime!(mr * nr * served * aw <= config.budget);
        let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);
        body::<E, EL, L, ER, V, A>(
            &mut acc_view,
            &lhs_mat,
            &rhs_mat,
            lw,
            served,
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
