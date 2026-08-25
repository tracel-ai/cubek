//! The general N-D gather nest, K-major with each operand read hoisted to its widest valid reuse
//! scope.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::instruction::registers::block;
use crate::*;

use super::{GatherProblem, coords::cell_read};

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and so the block's, `A` the
/// accumulator's.
#[cube]
pub(super) fn nest<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] problem: GatherProblem,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(problem.block.mr);
    let nr = comptime!(problem.block.nr);
    let cols = comptime!(problem.block.cols);
    let served = comptime!(problem.block.served);
    let spread = comptime!(problem.block.spread);
    let lw = comptime!(problem.block.lw);
    let aw = comptime!(problem.block.aw);
    let matrices = comptime!(problem.block.matrices());
    let batch_extents = comptime!(problem.block.batch_extents());
    let kc = comptime!(problem.block.kc);

    let lhs_view = lhs.nd_packed::<L>();
    let rhs_view = rhs.nd_packed::<V>();
    // Loop-invariant, and `comptime!`-bound so the `unroll` flag below stays a comptime binding:
    // `#[unroll(flag)]` silently rolls the loop when the macro cannot see `flag` as one.
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);

    // The fan-out walk names the lane with a comptime extract. Its final physical line can be
    // partial, just as the direct leaf's can, so retain a short tail rather than rejecting a
    // perfectly valid checked tile.
    let k_lines = comptime!(kc / lw);
    let k_tail = comptime!(kc % lw);

    let lane_fanout = comptime!(config.lane_fanout);
    let lane_index_exact = comptime!(problem.block.lane_index_exact());

    for mat in 0..matrices {
        let batch = unravel_const(comptime!(batch_extents.clone()), mat.fcast::<u32>());

        // The contraction's own algebra, as [`direct`](super::direct) states it.
        let mut acc = acc.matrix_accumulate::<A>(
            mat,
            comptime!(problem.block.space.clone()),
            comptime!(semiring.add()),
        );

        // Unroll only when no mask, otherwise compilation too long.
        let acc_check = acc.check();
        let unroll = comptime!(
            problem.block.scalars() <= config.budget && !lhs_check && !rhs_check && !acc_check
        );
        let mut c = block::seed::<E, V, A>(
            &mut acc,
            served,
            spread,
            aw,
            comptime!(mr),
            comptime!(nr),
            comptime!(cols),
            unroll,
        );

        // One rhs line per accumulator column, reused by every row of the rank-1 update. Held
        // across the whole K walk rather than re-declared per step, so the trace allocates it once
        // however many lane bodies the fan-out below emits. An rhs varying down the rows has no
        // such per-column value, and leaves this unwritten for the trace to fold away.
        let mut b = Array::<Vector<E, V>>::new(comptime!(nr));

        if comptime!(served > 1) {
            for step in 0..comptime!(kc / served) {
                rank1_update::<E, EL, L, ER, V>(
                    &lhs_view,
                    &rhs_view,
                    &mut c,
                    &mut b,
                    &batch,
                    step * comptime!(served),
                    comptime!(None),
                    unroll,
                    comptime!(problem.clone()),
                    semiring,
                );
            }
        } else if comptime!(lane_fanout && lw > 1 && lane_index_exact) {
            for line in 0..k_lines {
                #[unroll]
                for lane in 0..lw {
                    rank1_update::<E, EL, L, ER, V>(
                        &lhs_view,
                        &rhs_view,
                        &mut c,
                        &mut b,
                        &batch,
                        line * lw + lane,
                        comptime!(Some(lane)),
                        unroll,
                        comptime!(problem.clone()),
                        semiring,
                    );
                }
            }
            #[unroll]
            for lane in 0..k_tail {
                rank1_update::<E, EL, L, ER, V>(
                    &lhs_view,
                    &rhs_view,
                    &mut c,
                    &mut b,
                    &batch,
                    comptime!(k_lines * lw + lane),
                    comptime!(Some(lane)),
                    unroll,
                    comptime!(problem.clone()),
                    semiring,
                );
            }
        } else {
            // CPU and scalar lines keep the compact flat walk. Besides respecting the selected
            // configuration, this avoids cloning a wide fan-out body into LLVM IR when its fixed
            // extracts provide no benefit.
            for p in 0..kc {
                rank1_update::<E, EL, L, ER, V>(
                    &lhs_view,
                    &rhs_view,
                    &mut c,
                    &mut b,
                    &batch,
                    p,
                    comptime!(None),
                    unroll,
                    comptime!(problem.clone()),
                    semiring,
                );
            }
        }

        block::commit::<E, V, A>(
            &mut acc,
            c,
            served,
            spread,
            aw,
            comptime!(mr),
            comptime!(nr),
            comptime!(cols),
            unroll,
        );
    }
}

/// One gathered rank-1 update. `lane` names the component to take when the caller walks `K` as
/// (line, lane), so shader backends see a fixed `extract`; `None` is the flat walk, which resolves
/// the component from `reduce_coords` on the fastest contracted axis instead.
///
/// The span flags say which operand reads hoist out of the cell loop: each read is taken at the
/// coarsest cell the operand is invariant over, so the plain outer product still reads one lhs
/// per row and one rhs per column.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs_view: &MaskedView<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &MaskedView<'_, Vector<ER, V>, CoordsDyn>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    batch: &Coords<u32>,
    p: usize,
    #[comptime] lane: Option<usize>,
    #[comptime] unroll: bool,
    #[comptime] problem: GatherProblem,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(problem.block.mr);
    let nr = comptime!(problem.block.nr);
    let served = comptime!(problem.block.served);
    let lw = comptime!(problem.block.lw);
    let reduce_coords = unravel_const(
        comptime!(problem.block.reduce_extents.clone()),
        p.fcast::<u32>(),
    );

    // An rhs free of the row holds for every row, so its `nr` lines are read once here and reused
    // down the `i` loop.
    let k_axis_idx = comptime!(problem.block.reduce.len() - 1);
    if comptime!(!problem.rhs_spans_row) {
        #[unroll(unroll)]
        for n in 0..nr {
            b[n] = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                0u32,
                n as u32,
                &reduce_coords,
                comptime!(problem.rhs_space.clone()),
                comptime!(problem.clone()),
                comptime!(problem.block.vw),
            ));
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        // Whatever is invariant across the row's cells is read once here. Each stays at zero, and
        // folds away, when the cell loop reads that operand for itself.
        let mut a_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(!problem.lhs_spans_col) {
            // `resolve_nd_coords` divides the fastest contracted coordinate by `lw` into a line
            // index, so this is the same position for every lane of one line.
            let line = cell_read::<EL, L>(
                lhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(problem.lhs_space.clone()),
                comptime!(problem.clone()),
                lw,
            );
            a_row =
                lane_component::<E, EL, L, V>(line, &reduce_coords, lane, served, lw, k_axis_idx);
        }
        let mut b_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(problem.rhs_spans_row && !problem.rhs_spans_col) {
            b_row = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(problem.rhs_space.clone()),
                comptime!(problem.clone()),
                comptime!(problem.block.vw),
            ));
        }
        #[unroll(unroll)]
        for n in 0..nr {
            let a = if comptime!(problem.lhs_spans_col) {
                let line = cell_read::<EL, L>(
                    lhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(problem.lhs_space.clone()),
                    comptime!(problem.clone()),
                    lw,
                );
                lane_component::<E, EL, L, V>(line, &reduce_coords, lane, served, lw, k_axis_idx)
            } else {
                a_row
            };
            let v = if comptime!(!problem.rhs_spans_row) {
                b[n]
            } else if comptime!(!problem.rhs_spans_col) {
                b_row
            } else {
                Vector::<E, V>::cast_from(cell_read::<ER, V>(
                    rhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(problem.rhs_space.clone()),
                    comptime!(problem.clone()),
                    comptime!(problem.block.vw),
                ))
            };
            // One semiring step, for the reason [`block::rank1_update`] gives.
            c[i * nr + n] = semiring.step::<Vector<E, V>>(a, v, c[i * nr + n]);
        }
    }
}

/// The `K` component of one lhs line, widened into the accumulate element. The whole line at a
/// folded step, fixed when the caller walks `K` as (line, lane), resolved from the fastest
/// contracted coordinate in `reduce_coords` on the flat walk.
#[cube]
fn lane_component<E: Numeric, EL: Numeric, L: Size, V: Size>(
    line: Vector<EL, L>,
    reduce_coords: &Coords<u32>,
    #[comptime] lane: Option<usize>,
    #[comptime] served: usize,
    #[comptime] lw: usize,
    #[comptime] k_axis_idx: usize,
) -> Vector<E, V> {
    if comptime!(served > 1) {
        Vector::<E, V>::cast_from(line)
    } else if comptime!(lane.is_some()) {
        Vector::<E, V>::cast_from(line.extract(comptime!(lane.unwrap())))
    } else if comptime!(lw == 1) {
        Vector::<E, V>::cast_from(line.extract(0usize))
    } else {
        let last_k = reduce_coords.at(comptime!(k_axis_idx));
        Vector::<E, V>::cast_from(
            line.extract_dynamic((last_k % comptime!(lw as u32)).fcast::<usize>()),
        )
    }
}
