//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use crate::instruction::registers::block;
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
    comptime!(assert!(
        rw == aw || served > 1,
        "contract direct: a padded rhs staged wider than its {aw}-wide sink must use the N-D nest"
    ));

    // The block's lines are the rhs's: `served`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(served > 1) {
        let size!(W) = served;
        let size!(A) = 1usize;
        nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, space, served, lw, 1usize, config, semiring);
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest::<E, EL, W, ER, A, A>(acc, lhs, rhs, space, served, lw, aw, config, semiring);
    }
}

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and so the block's, `A` the
/// accumulator's.
#[cube]
#[allow(clippy::too_many_arguments)]
fn nest<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] served: usize,
    #[comptime] lw: usize,
    #[comptime] aw: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let rank = comptime!(space.rank());
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let k = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space)[0]);
    let kc = comptime!(merged.extent(k));

    // `nr` counts the accumulator's own lines along `N`; `mr` (rows) and `kc` (scalar `K`) are
    // unvectorized. `cols` is the scalar extent behind `nr`, which the block only consults on the
    // N-D nest's spread path.
    let cols = comptime!(space.extent_at(rank - 1));
    let (mr, nr) = comptime!((space.extent_at(rank - 2), cols / aw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());

    // Only the bound proof below needs the lhs's line count; the walk itself splits `kc`.
    let lhs_k_lines = comptime!(kc.div_ceil(lw));
    let lane_fanout = comptime!(config.lane_fanout);

    for mat in 0..matrices {
        let lhs_mat = lhs.matrix_packed::<L>(mat);
        let rhs_mat = rhs.matrix_packed::<V>(mat);
        // The contraction's own algebra: its products accumulate under the semiring's add.
        let mut acc_view =
            acc.matrix_accumulate::<A>(mat, comptime!(space.clone()), comptime!(semiring.add()));

        // A checked edge normally rolls every local array access. When enabled, split the leaf
        // into two comptime-specialized bodies: interior instances prove their complete operand
        // and accumulator blocks in bounds once, then keep `c` and `b` in registers; only the
        // actual edge instance pays masked reads/writes and runtime local-array indexing.
        let lhs_check = comptime!(lhs_mat.check);
        let rhs_check = comptime!(rhs_mat.check);
        let acc_check = acc_view.check();
        // The budget is scalars, and the block holds `mr * nr` lines of `served * aw` (exactly one
        // of the two exceeds 1).
        let eligible = comptime!(mr * nr * served * aw <= config.budget);
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
