//! The N-D contraction nest: multiple contracted axes, or projected operands.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::instruction::registers::block;
use crate::*;

/// N-D variant of [`direct::contract`](super::direct::contract) for operations with
/// multiple contracted axes or projected operands.
///
/// The outer product it runs is an optimization, not the definition of the contraction: it holds
/// only while the lhs is free of the accumulator's column and the rhs free of its row, which is
/// what lets one read of each serve a whole row or column of cells. A resampling operand breaks
/// that: the gather makes the value depend on the output position, and a procedural weight
/// depends on it too, so an operand spanning the other's free axis is read at the cell instead,
/// and the rank-1 update degenerates to a per-cell `fma`. Only the reads change; what is
/// contracted, and the accumulator block living in registers across `kc`, do not.
#[cube]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] served: usize,
    #[comptime] config: RegisterBlock,
) {
    let lw = lhs.vector_size();
    let aw = comptime!(acc.store.vector_size);

    // The block's lines are the rhs's: `served`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(served > 1) {
        let size!(W) = served;
        let size!(A) = 1usize;
        nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, space, served, lw, served, 1usize, config);
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest::<E, EL, W, ER, A, A>(acc, lhs, rhs, space, served, lw, aw, aw, config);
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
    #[comptime] vw: usize,
    #[comptime] aw: usize,
    #[comptime] config: RegisterBlock,
) {
    let rank = comptime!(space.rank());
    let (mr, nr) = comptime!((space.extent_at(rank - 2), space.extent_at(rank - 1) / aw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());

    // The reduce axes' extents come off the operands' merged space, not the accumulator's: a
    // contracted axis is by definition absent from the accumulator, and an axis only one operand
    // spans still has to be walked.
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let reduce = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space).to_vec());
    let reduce_extents = comptime!(reduce.iter().map(|&a| merged.extent(a)).collect::<Vec<_>>());
    let kc = comptime!(reduce_extents.iter().product::<usize>());

    // Whether either operand varies along the axis the outer product assumes it is free of.
    // `rhs_spans_col` separates an rhs that merely varies down the rows, and so still holds for a
    // whole row of cells, from one that varies cell by cell.
    let lhs_spans_col = comptime!(lhs.space.contains(space.axis_at(rank - 1)));
    let rhs_spans_row = comptime!(rhs.space.contains(space.axis_at(rank - 2)));
    let rhs_spans_col = comptime!(rhs.space.contains(space.axis_at(rank - 1)));

    comptime!(assert_operand_shapes(
        &lhs.space,
        &rhs.space,
        &space,
        &reduce,
        vw,
        lhs_spans_col
    ));

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

    for mat in 0..matrices {
        let batch = unravel(
            &const_coords(comptime!(
                (0..rank - 2)
                    .map(|p| space.extent_at(p))
                    .collect::<Vec<_>>()
            )),
            mat.fcast::<u32>(),
        );

        let mut acc = acc.matrix_accumulate::<A>(mat, comptime!(space.clone()));

        // Unroll only when no mask, otherwise compilation too long.
        let acc_check = acc.check();
        // The budget is scalars, and the block holds `mr * nr` lines of `served * aw` (exactly
        // one of the two exceeds 1).
        let unroll = comptime!(
            mr * nr * served * aw <= config.budget && !lhs_check && !rhs_check && !acc_check
        );
        let mut c = block::seed::<E, V, A>(&mut acc, served, comptime!(mr), comptime!(nr), unroll);

        // One rhs line per accumulator column, reused by every row of the rank-1 update. Held
        // across the whole K walk rather than re-declared per step, so the trace allocates it once
        // however many lane bodies the fan-out below emits. An rhs varying down the rows has no
        // such per-column value, and leaves this unwritten for the trace to fold away.
        let mut b = Array::<Vector<E, V>>::new(nr);

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
                    served,
                    mr,
                    nr,
                    unroll,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lw,
                    vw,
                    lhs_spans_col,
                    rhs_spans_row,
                    rhs_spans_col,
                );
            }
        } else if comptime!(lane_fanout && lw > 1) {
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
                        served,
                        mr,
                        nr,
                        unroll,
                        comptime!(space.clone()),
                        comptime!(reduce.clone()),
                        comptime!(reduce_extents.clone()),
                        comptime!(lhs.space.clone()),
                        comptime!(rhs.space.clone()),
                        lw,
                        vw,
                        lhs_spans_col,
                        rhs_spans_row,
                        rhs_spans_col,
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
                    served,
                    mr,
                    nr,
                    unroll,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lw,
                    vw,
                    lhs_spans_col,
                    rhs_spans_row,
                    rhs_spans_col,
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
                    served,
                    mr,
                    nr,
                    unroll,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lw,
                    vw,
                    lhs_spans_col,
                    rhs_spans_row,
                    rhs_spans_col,
                );
            }
        }

        block::commit::<E, V, A>(&mut acc, c, served, comptime!(mr), comptime!(nr), unroll);
    }
}

/// One gathered rank-1 update. `lane` names the component to take when the caller walks `K` as
/// (line, lane), so shader backends see a fixed `extract`; `None` is the flat walk, which resolves
/// the component from `p` instead.
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
    #[comptime] served: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
    #[comptime] space: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] reduce_extents: Vec<usize>,
    #[comptime] lhs_space: Space,
    #[comptime] rhs_space: Space,
    #[comptime] lw: usize,
    #[comptime] vw: usize,
    #[comptime] lhs_spans_col: bool,
    #[comptime] rhs_spans_row: bool,
    #[comptime] rhs_spans_col: bool,
) {
    let reduce_coords = unravel(&const_coords(reduce_extents), p.fcast::<u32>());

    // An rhs free of the row holds for every row, so its `nr` lines are read once here and reused
    // down the `i` loop.
    if comptime!(!rhs_spans_row) {
        #[unroll(unroll)]
        for n in 0..nr {
            b[n] = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                0u32,
                n as u32,
                &reduce_coords,
                comptime!(rhs_space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                vw,
            ));
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        // Whatever is invariant across the row's cells is read once here. Each stays at zero, and
        // folds away, when the cell loop reads that operand for itself.
        let mut a_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(!lhs_spans_col) {
            // `resolve_nd_coords` divides the fastest contracted coordinate by `lw` into a line
            // index, so this is the same position for every lane of one line.
            let line = cell_read::<EL, L>(
                lhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(lhs_space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                lw,
            );
            a_row = lane_component::<E, EL, L, V>(line, p, lane, served, lw);
        }
        let mut b_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(rhs_spans_row && !rhs_spans_col) {
            b_row = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(rhs_space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                vw,
            ));
        }
        #[unroll(unroll)]
        for n in 0..nr {
            let a = if comptime!(lhs_spans_col) {
                let line = cell_read::<EL, L>(
                    lhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(lhs_space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    lw,
                );
                lane_component::<E, EL, L, V>(line, p, lane, served, lw)
            } else {
                a_row
            };
            let v = if comptime!(!rhs_spans_row) {
                b[n]
            } else if comptime!(!rhs_spans_col) {
                b_row
            } else {
                Vector::<E, V>::cast_from(cell_read::<ER, V>(
                    rhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(rhs_space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    vw,
                ))
            };
            // One semiring step, for the reason [`block::rank1_update`] gives.
            c[i * nr + n] = Semiring::SUM_PROD.step::<Vector<E, V>>(a, v, c[i * nr + n]);
        }
    }
}

/// One operand read at the accumulator cell `(row, col)` of the batch matrix `batch` names.
/// `width` is the operand's own line width, since only its innermost axis is addressed in lines.
#[cube]
fn cell_read<T: Numeric, W: Size>(
    view: &MaskedView<'_, Vector<T, W>, CoordsDyn>,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] operand: Space,
    #[comptime] acc: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] width: usize,
) -> Vector<T, W> {
    let acc_coords = acc_cell_coords(batch, row, col);
    let pos = resolve_nd_coords(
        operand,
        acc,
        reduce,
        &acc_coords,
        reduce_coords,
        width,
        false,
    );
    view.read(pos)
}

/// The `K` component of one lhs line, widened into the accumulate element. The whole line at a
/// folded step, fixed when the caller walks `K` as (line, lane), resolved from `p` on the flat
/// walk.
#[cube]
fn lane_component<E: Numeric, EL: Numeric, L: Size, V: Size>(
    line: Vector<EL, L>,
    p: usize,
    #[comptime] lane: Option<usize>,
    #[comptime] served: usize,
    #[comptime] lw: usize,
) -> Vector<E, V> {
    if comptime!(served > 1) {
        Vector::<E, V>::cast_from(line)
    } else if comptime!(lane.is_some()) {
        Vector::<E, V>::cast_from(line.extract(comptime!(lane.unwrap())))
    } else if comptime!(lw == 1) {
        Vector::<E, V>::cast_from(line.extract(0usize))
    } else {
        Vector::<E, V>::cast_from(line.extract_dynamic(p % lw))
    }
}

/// Assembles the accumulator cell coordinate [`resolve_nd_coords`] reads on its acc branch:
/// `batch`'s own axes in order, then `row` (the accumulator's second to last axis), then `col`
/// (its last). This is the axis order [`resolve_nd_coords`] assumes when it looks up
/// `acc.position(axis)`.
#[cube]
fn acc_cell_coords(batch: &Coords<u32>, row: u32, col: u32) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..batch.len() {
        out.push(batch.at(p));
    }
    out.push(row);
    out.push(col);

    out
}

/// What [`resolve_nd_coords`] and the lane fold assume about how the operands are lined up. Both
/// treat one axis per operand as the vectorized one and address it in lines; if that is not the
/// axis the operand actually lines along, the reads are silently off by the width rather than
/// wrong in a way a test would localize. Host-side, so a violation is a comptime message.
fn assert_operand_shapes(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    reduce: &[Axis],
    rhs_vec_len: usize,
    lhs_spans_col: bool,
) {
    assert!(
        !reduce.is_empty(),
        "contract gather: the operands contract no axis against the accumulator"
    );
    // `Space::contracted` merges lhs-first, so an axis only the rhs spans lands past every lhs
    // one and would take the `fastest` slot below. That reads as the lhs being lined wrong, which
    // it is not, so the real constraint is named here instead.
    for &axis in reduce {
        assert!(
            lhs.contains(axis),
            "contract gather: the lhs must span every contracted axis, but {axis:?} is contracted by \
             the rhs alone"
        );
    }
    let fastest = reduce[reduce.len() - 1];
    assert!(
        lhs.axis_at(lhs.rank() - 1) == fastest,
        "contract gather: the lhs must line along the fastest contracted axis {fastest:?}"
    );
    // A vectorized rhs lines along the accumulator's innermost axis (its lanes are cells) or the
    // fastest contracted one (its lanes are partials of one cell); a scalar one is addressed in
    // elements and need not span either, which is what lets a weight shared by every column live
    // in a space that simply omits it.
    let rhs_lined = rhs.axis_at(rhs.rank() - 1);
    assert!(
        rhs_vec_len == 1 || rhs_lined == acc.axis_at(acc.rank() - 1) || rhs_lined == fastest,
        "contract gather: a vectorized rhs must line along the accumulator's innermost axis or \
         the fastest contracted axis {fastest:?}"
    );
    // An lhs varying along the column is read once per cell, and a cell is `rhs_vec_len` columns
    // wide. The lhs lines along a contracted axis, not this one, so one read cannot cover them:
    // it would need a value per lane off an axis it does not line along, and the broadcast would
    // silently serve the first column's value to all of them.
    assert!(
        !lhs_spans_col || rhs_vec_len == 1,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis cannot also be served in lines (the accumulator is {rhs_vec_len} \
         wide)"
    );
}
