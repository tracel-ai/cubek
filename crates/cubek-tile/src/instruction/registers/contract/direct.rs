//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use crate::instruction::registers::block;
use crate::*;

/// The contraction nest for a fixed lhs (`IL`) and rhs (`IR`) storage element: over each batch
/// matrix, the `mr × nr` block of `V`-wide accumulators lives in registers (load once, `kc` rank-1
/// updates, store once). `pack_l`/`pack_r` narrow each operand's physical line (`served / pack`,
/// `1` for plain/native). The storage element per operand is the price of a typed quant view
/// (`#[cube]` takes no `impl Trait`, so the view can't be erased behind a `read` trait); `#[cube]`
/// inlines at trace time, so folding the rank-1 step in here costs nothing over a separate fn.
///
/// The 2-D form its reads assume: `mat` indexes a batch matrix, `(row, k)` and `(k, col)` address
/// the operands. [`memory`](super::memory) routes anything else to
/// [`gather::contract`](super::gather::contract), so the two conditions below are
/// re-asserted rather than re-decided.
#[cube]
pub(super) fn contract<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    V: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] pack_l: usize,
    #[comptime] pack_r: usize,
    #[comptime] config: RegisterBlock,
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

    // `nr` is a line count (spans `N` in `V`-wide lines); `mr` (rows) and `kc` (scalar `K`, off
    // `rhs`) are unvectorized. A packed operand's physical line is `served / pack` narrower.
    let lw = lhs.vector_size();
    let vw = rhs.vector_size();
    let (mr, nr, kc) = comptime! {
        (
            space.extent_at(space.rank() - 2),
            space.extent_at(space.rank() - 1) / vw,
            rhs.space.extent_at(rhs.space.rank() - 2)
        )
    };
    let size!(WPL) = comptime!(lw / pack_l);
    let size!(WPR) = comptime!(vw / pack_r);

    let matrices = comptime! {
        let mut count = 1;
        for p in 0..space.rank() - 2 {
            count *= space.extent_at(p);
        }
        count
    };

    // Only the bound proof below needs the lhs's line count; the walk itself splits `kc`.
    let lhs_k_lines = comptime!(kc.div_ceil(lw));

    // The budget is scalars; `nr` counts `vw`-wide lines, so compare in scalars.
    let lane_fanout = comptime!(config.lane_fanout);

    for mat in 0..matrices {
        let lhs = lhs.matrix_transparent::<IL, WPL, L>(mat);
        let rhs = rhs.matrix_transparent::<IR, WPR, V>(mat);
        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));

        // A checked edge normally rolls every local array access. When enabled, split the leaf
        // into two comptime-specialized bodies: interior instances prove their complete operand
        // and accumulator blocks in bounds once, then keep `c` and `b` in registers; only the
        // actual edge instance pays masked reads/writes and runtime local-array indexing.
        let lhs_check = comptime!(lhs.check);
        let rhs_check = comptime!(rhs.check);
        let acc_check = acc.check();
        let eligible = comptime!(mr * nr * vw <= config.budget);
        let split_edge =
            comptime!(eligible && config.split_edge && (lhs_check || rhs_check || acc_check));
        if comptime!(split_edge) {
            let origin = (0u32.runtime(), 0u32.runtime());
            let lhs_extent = (
                comptime!(mr as u32).runtime(),
                comptime!(lhs_k_lines as u32).runtime(),
            );
            let rhs_extent = (
                comptime!(kc as u32).runtime(),
                comptime!(nr as u32).runtime(),
            );
            let acc_extent = (
                comptime!(mr as u32).runtime(),
                comptime!(nr as u32).runtime(),
            );
            let in_bounds = lhs.block_in_bounds(origin, lhs_extent)
                && rhs.block_in_bounds(origin, rhs_extent)
                && acc.block_in_bounds(origin, acc_extent);
            if in_bounds {
                contract_body::<E, EL, L, ER, V>(
                    &mut acc,
                    &lhs,
                    &rhs,
                    lw,
                    mr,
                    nr,
                    kc,
                    true,
                    lane_fanout,
                );
            } else {
                contract_body::<E, EL, L, ER, V>(
                    &mut acc,
                    &lhs,
                    &rhs,
                    lw,
                    mr,
                    nr,
                    kc,
                    false,
                    lane_fanout,
                );
            }
        } else {
            let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);
            contract_body::<E, EL, L, ER, V>(
                &mut acc,
                &lhs,
                &rhs,
                lw,
                mr,
                nr,
                kc,
                unroll,
                lane_fanout,
            );
        }
    }
}

/// The complete 2-D nest body, specialized at trace time for either register-resident local
/// arrays (`unroll = true`) or the checked edge fallback (`unroll = false`).
#[cube]
fn contract_body<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    #[comptime] lw: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
) {
    let mut c = block::seed(acc, mr, nr, unroll);
    block::contract::<E, EL, L, ER, V>(lhs, rhs, &mut c, lw, mr, nr, kc, unroll, lane_fanout);
    block::commit(acc, c, mr, nr, unroll);
}
