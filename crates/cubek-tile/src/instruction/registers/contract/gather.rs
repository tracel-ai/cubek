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
    #[comptime] semiring: Semiring,
) {
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    // `step_served` only returns 1 for an rhs lining along the accumulator, where it already
    // refused anything but a matched pair or a scalar sink, so the division is exact.
    comptime!(assert!(
        rw == aw || aw == 1,
        "contract gather: a rhs staged wider than its sink spreads its lanes across scalar cells, \
         so the accumulator must be served scalar (rhs {rw}, accumulator {aw})"
    ));
    let spread = comptime!(if served > 1 { 1usize } else { rw / aw });

    // The block's lines are the rhs's: `served`-wide K-partials of one cell at a folded step,
    // `aw`-wide neighbouring cells otherwise.
    if comptime!(served > 1) {
        let size!(W) = served;
        let size!(A) = 1usize;
        nest::<E, EL, W, ER, W, A>(
            acc, lhs, rhs, space, served, spread, lw, served, 1usize, config, semiring,
        );
    } else {
        let size!(W) = lw;
        let size!(V) = rw;
        let size!(A) = aw;
        nest::<E, EL, W, ER, V, A>(
            acc, lhs, rhs, space, served, spread, lw, rw, aw, config, semiring,
        );
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
    #[comptime] spread: usize,
    #[comptime] lw: usize,
    #[comptime] vw: usize,
    #[comptime] aw: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let rank = comptime!(space.rank());
    // `cols` is the sink's own innermost extent. Only a spread block rounds up: its lanes are
    // scalar cells, so a last column overhanging `cols` is masked lane by lane in
    // [`block::seed`]/[`block::commit`]. An `aw`-wide column has no such handle on a partial
    // cell, and keeps counting whole lines.
    let cols = comptime!(space.extent_at(rank - 1));
    let nr = comptime!(if spread > 1 {
        cols.div_ceil(spread)
    } else {
        cols / aw
    });
    let mr = comptime!(space.extent_at(rank - 2));
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

    // An lhs lined along the accumulator's innermost axis rather than along `K`. The case a
    // batched contraction needs: an axis every operand spans — a depthwise convolution's channel
    // — is the accumulator's column, and the lhs holds one value per column of it rather than one
    // value per row. Its line then *is* the cell, so there is no `K` component to extract.
    let lhs_lines_col = comptime!(
        lhs_spans_col && lhs.space.axis_at(lhs.space.rank() - 1) == space.axis_at(rank - 1)
    );

    comptime!(assert_operand_shapes(
        &lhs.space,
        &rhs.space,
        &space,
        &reduce,
        vw,
        lw,
        lhs_spans_col,
        lhs_lines_col,
    ));

    let lhs_view = lhs.nd_packed::<L>(comptime!(Guard::Checked));
    let rhs_view = rhs.nd_packed::<V>(comptime!(Guard::Checked));
    // Loop-invariant, and `comptime!`-bound so the `unroll` flag below stays a comptime binding:
    // `#[unroll(flag)]` silently rolls the loop when the macro cannot see `flag` as one.
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);

    // The fan-out walk names the lane with a comptime extract. Its final physical line can be
    // partial, just as the direct leaf's can, so retain a short tail rather than rejecting a
    // perfectly valid checked tile. Only a single contracted axis ever reaches that tail: past
    // one, the gate below already asks that the fastest axis is a whole number of lines.
    let k_lines = comptime!(kc / lw);
    let k_tail = comptime!(kc % lw);

    let lane_fanout = comptime!(config.lane_fanout);
    // The fixed extract the fan-out names has to agree with the coordinate `lane_component`
    // decodes on the flat walk, `last_k % lw`. A single contracted axis makes the two the same
    // expression, whatever its extent: `last_k` *is* the flat step. Past one, `last_k` restarts
    // every `reduce_extents.last()` steps, so the flat index only stays in step with it when that
    // extent is a whole number of lines (a padded stage's is not).
    let lane_index_exact =
        comptime!(reduce.len() == 1 || reduce_extents[reduce_extents.len() - 1].is_multiple_of(lw));

    // The block only lives in registers while it fits the budget, and a rolled block gains
    // nothing from an unguarded view: the split is worth its second copy of the walk only when
    // the fast side would actually unroll.
    let eligible = comptime!(mr * nr * served * aw * spread <= config.budget);
    // Not every guard is a redundant zero mask the corner check below can retire; a clamp is
    // the one it cannot. `Tile::guard_provable` is where that list lives.
    let lhs_provable = lhs.guard_provable();
    let rhs_provable = rhs.guard_provable();
    let provable = comptime!(lhs_provable && rhs_provable);
    // A spread block rounds `nr` up, so its last column addresses a line past the operands' own
    // extent — one past the far corner [`box_in_bounds`] proves, which counts whole lines. The
    // accumulator has [`block::seed`]/[`block::commit`]'s per-lane mask for those spare lanes; an
    // operand read that has dropped its guard has nothing, so keep the whole leaf checked.
    let spread_overhang = comptime!(block::spread_guard(spread, cols));
    let split_operands = comptime!(
        config.split_edge && eligible && provable && !spread_overhang && (lhs_check || rhs_check)
    );
    // Whether the operands' whole boxes are inside their buffers. Hoisted out of the matrix loop
    // because it is a statement about the operand, not about which batch matrix is being read,
    // and computed only when something below would act on it. `L` is the lhs's line width as the
    // view reads it, which at a folded step is `served` rather than the tile's own.
    let l_lines = comptime!(match served > 1 {
        true => served,
        false => lw,
    });
    let operands_inside = if comptime!(split_operands) {
        box_in_bounds::<EL, L>(&lhs_view, comptime!(lhs.space.clone()), l_lines)
            && box_in_bounds::<ER, V>(&rhs_view, comptime!(rhs.space.clone()), vw)
    } else {
        comptime!(false).runtime()
    };

    for mat in 0..matrices {
        let batch = unravel(
            &const_coords(comptime!(
                (0..rank - 2)
                    .map(|p| space.extent_at(p))
                    .collect::<Vec<_>>()
            )),
            mat.fcast::<u32>(),
        );

        // The contraction's own algebra, as [`direct`](super::direct) states it.
        let mut acc =
            acc.matrix_accumulate::<A>(mat, comptime!(space.clone()), comptime!(semiring.add()));

        // Unroll only when no mask, otherwise compilation too long.
        let acc_check = acc.check();
        let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);

        // A checked operand rolls the whole walk: every read re-proves its own bounds, and the
        // block cannot live in registers because its indices stop being comptime. Splitting the
        // leaf is what stops an interior instance paying for the edge's guard — it proves the
        // whole box once and then reads through views with no guard left in them, while the
        // instances that really do straddle an edge take the masked walk unchanged.
        //
        // The *operands* only. The accumulator keeps its guard on both sides: it is written once
        // per cell against `kc` operand reads per cell, so dropping its guard buys a fraction of
        // a percent, and what it would cost is the one thing a leaf must never get wrong — a
        // write landing outside the output.
        if comptime!(split_operands) {
            let inside = operands_inside;
            if inside {
                walk::<E, EL, L, ER, V, A>(
                    &lhs.nd_packed::<L>(comptime!(Guard::Proved)),
                    &rhs.nd_packed::<V>(comptime!(Guard::Proved)),
                    &mut acc,
                    &batch,
                    served,
                    spread,
                    aw,
                    cols,
                    mr,
                    nr,
                    kc,
                    k_lines,
                    k_tail,
                    lw,
                    vw,
                    // Every read on this side is proved in bounds, so the block's indices stay
                    // comptime and it can live in registers whatever the accumulator's guard is.
                    comptime!(true),
                    lane_fanout,
                    lane_index_exact,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lhs_spans_col,
                    lhs_lines_col,
                    rhs_spans_row,
                    rhs_spans_col,
                    semiring,
                );
            } else {
                walk::<E, EL, L, ER, V, A>(
                    &lhs_view,
                    &rhs_view,
                    &mut acc,
                    &batch,
                    served,
                    spread,
                    aw,
                    cols,
                    mr,
                    nr,
                    kc,
                    k_lines,
                    k_tail,
                    lw,
                    vw,
                    comptime!(false),
                    lane_fanout,
                    lane_index_exact,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lhs_spans_col,
                    lhs_lines_col,
                    rhs_spans_row,
                    rhs_spans_col,
                    semiring,
                );
            }
        } else {
            walk::<E, EL, L, ER, V, A>(
                &lhs_view,
                &rhs_view,
                &mut acc,
                &batch,
                served,
                spread,
                aw,
                cols,
                mr,
                nr,
                kc,
                k_lines,
                k_tail,
                lw,
                vw,
                unroll,
                lane_fanout,
                lane_index_exact,
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                comptime!(reduce_extents.clone()),
                comptime!(lhs.space.clone()),
                comptime!(rhs.space.clone()),
                lhs_spans_col,
                lhs_lines_col,
                rhs_spans_row,
                rhs_spans_col,
                semiring,
            );
        }
    }
}

/// Whether every read the walk will take through `view` lands inside it.
///
/// The two extreme corners of the operand's box are enough. A [`Projection`] scales each logical
/// coordinate by a non-negative factor and adds a constant, so the physical coordinate it yields
/// is monotone in every logical one: nothing between the corners can reach outside what they
/// bracket. `is_in_bounds` composes down the whole view stack, so one call covers the logical
/// extents, the window's own bound (which is where padding shows up), and the buffer.
///
/// The far corner is the operand's own extent in *whole* lines, so this proves only the reads a
/// walk staying inside that box takes. A caller whose column count overhangs the extent — the
/// spread block's rounded-up `nr` — reaches a line this never looked at, and must not act on a
/// `true` from here.
#[cube]
#[allow(clippy::needless_range_loop)]
fn box_in_bounds<T: Numeric, W: Size>(
    view: &MaskedView<'_, Vector<T, W>, CoordsDyn>,
    #[comptime] space: Space,
    #[comptime] width: usize,
) -> bool {
    let rank = comptime!(space.rank());
    let extents = comptime!(crate::line_extents(&space, width, 0, rank));

    let mut near = CoordsDyn::new();
    let mut far = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        near.push(0u32.runtime());
        far.push(comptime!(extents[p] as u32 - 1).runtime());
    }

    view.is_in_bounds(near) && view.is_in_bounds(far)
}

/// The `K` walk over one batch matrix: seed the `mr × nr` block, fold `kc` rank-1 updates into
/// it, commit it back. Split out of [`nest`] so the same walk serves both sides of the edge
/// split, differing only in the views handed to it and whether it unrolls.
#[cube]
#[allow(clippy::too_many_arguments)]
fn walk<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    lhs_view: &MaskedView<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &MaskedView<'_, Vector<ER, V>, CoordsDyn>,
    acc: &mut AccumulateView<'_, E, A>,
    batch: &Coords<u32>,
    #[comptime] served: usize,
    #[comptime] spread: usize,
    #[comptime] aw: usize,
    #[comptime] cols: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] k_lines: usize,
    #[comptime] k_tail: usize,
    #[comptime] lw: usize,
    #[comptime] vw: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
    #[comptime] lane_index_exact: bool,
    #[comptime] space: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] reduce_extents: Vec<usize>,
    #[comptime] lhs_space: Space,
    #[comptime] rhs_space: Space,
    #[comptime] lhs_spans_col: bool,
    #[comptime] lhs_lines_col: bool,
    #[comptime] rhs_spans_row: bool,
    #[comptime] rhs_spans_col: bool,
    #[comptime] semiring: Semiring,
) {
    let mut c = block::seed::<E, V, A>(
        acc,
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
                lhs_view,
                rhs_view,
                &mut c,
                &mut b,
                batch,
                step * comptime!(served),
                comptime!(None),
                served,
                mr,
                nr,
                unroll,
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                comptime!(reduce_extents.clone()),
                comptime!(lhs_space.clone()),
                comptime!(rhs_space.clone()),
                lw,
                vw,
                lhs_spans_col,
                lhs_lines_col,
                rhs_spans_row,
                rhs_spans_col,
                semiring,
            );
        }
    } else if comptime!(lane_fanout && lw > 1 && !lhs_lines_col && lane_index_exact) {
        for line in 0..k_lines {
            #[unroll]
            for lane in 0..lw {
                rank1_update::<E, EL, L, ER, V>(
                    lhs_view,
                    rhs_view,
                    &mut c,
                    &mut b,
                    batch,
                    line * lw + lane,
                    comptime!(Some(lane)),
                    served,
                    mr,
                    nr,
                    unroll,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs_space.clone()),
                    comptime!(rhs_space.clone()),
                    lw,
                    vw,
                    lhs_spans_col,
                    lhs_lines_col,
                    rhs_spans_row,
                    rhs_spans_col,
                    semiring,
                );
            }
        }
        #[unroll]
        for lane in 0..k_tail {
            rank1_update::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut c,
                &mut b,
                batch,
                comptime!(k_lines * lw + lane),
                comptime!(Some(lane)),
                served,
                mr,
                nr,
                unroll,
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                comptime!(reduce_extents.clone()),
                comptime!(lhs_space.clone()),
                comptime!(rhs_space.clone()),
                lw,
                vw,
                lhs_spans_col,
                lhs_lines_col,
                rhs_spans_row,
                rhs_spans_col,
                semiring,
            );
        }
    } else {
        // CPU and scalar lines keep the compact flat walk. Besides respecting the selected
        // configuration, this avoids cloning a wide fan-out body into LLVM IR when its fixed
        // extracts provide no benefit.
        #[unroll(unroll)]
        for p in 0..kc {
            rank1_update::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut c,
                &mut b,
                batch,
                p,
                comptime!(None),
                served,
                mr,
                nr,
                unroll,
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                comptime!(reduce_extents.clone()),
                comptime!(lhs_space.clone()),
                comptime!(rhs_space.clone()),
                lw,
                vw,
                lhs_spans_col,
                lhs_lines_col,
                rhs_spans_row,
                rhs_spans_col,
                semiring,
            );
        }
    }

    block::commit::<E, V, A>(
        acc,
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
    #[comptime] lhs_lines_col: bool,
    #[comptime] rhs_spans_row: bool,
    #[comptime] rhs_spans_col: bool,
    #[comptime] semiring: Semiring,
) {
    let reduce_coords = unravel(&const_coords(reduce_extents), p.fcast::<u32>());

    // An rhs free of the row holds for every row, so its `nr` lines are read once here and reused
    // down the `i` loop.
    let k_axis_idx = comptime!(reduce.len() - 1);
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
            a_row =
                lane_component::<E, EL, L, V>(line, &reduce_coords, lane, served, lw, k_axis_idx);
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
                // A col-lined lhs addresses its innermost axis in lines, exactly as the rhs
                // does, and the line it reads is the cell: every column of it is a different
                // value, which is the whole point of lining along that axis.
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
                if comptime!(lhs_lines_col) {
                    Vector::<E, V>::cast_from(line)
                } else {
                    lane_component::<E, EL, L, V>(
                        line,
                        &reduce_coords,
                        lane,
                        served,
                        lw,
                        k_axis_idx,
                    )
                }
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
            c[i * nr + n] = semiring.step::<Vector<E, V>>(a, v, c[i * nr + n]);
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
#[allow(clippy::too_many_arguments)]
fn assert_operand_shapes(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    reduce: &[Axis],
    rhs_vec_len: usize,
    lhs_vec_len: usize,
    lhs_spans_col: bool,
    lhs_lines_col: bool,
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
    // A col-lined lhs is the exception: it lines along the accumulator's innermost axis, so the
    // fastest contracted axis is walked in elements like every other contracted one.
    assert!(
        lhs_lines_col || lhs.axis_at(lhs.rank() - 1) == fastest,
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
    // wide. One read covers them only if that is the axis it lines along — which is what
    // `lhs_lines_col` says. Lined along a contracted axis instead, it would need a value per lane
    // off an axis it does not line along, and the broadcast would silently serve the first
    // column's value to all of them.
    assert!(
        !lhs_spans_col || rhs_vec_len == 1 || lhs_lines_col,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis must be the one it lines along (the accumulator is {rhs_vec_len} \
         wide, the lhs lines {lhs_vec_len})"
    );
    // The col-lined line *is* the cell, so the two are the same width by construction.
    assert!(
        !lhs_lines_col || lhs_vec_len == rhs_vec_len,
        "contract gather: a col-lined lhs is read as the cell itself, so its line width \
         ({lhs_vec_len}) must be the accumulator's ({rhs_vec_len})"
    );
}
