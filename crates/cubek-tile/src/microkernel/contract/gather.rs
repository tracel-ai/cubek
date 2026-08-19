//! The N-D contraction nest: multiple contracted axes, or projected operands.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::microkernel::block;
use crate::*;

/// N-D variant of [`direct::contract`](super::direct::contract) for operations with
/// multiple contracted axes or projected operands.
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
    #[comptime] config: MemoryMmaConfig,
) {
    let lw = lhs.vector_size();
    let vw = rhs.vector_size();
    let size!(WPL) = comptime!(lw / pack_l);
    let size!(WPR) = comptime!(vw / pack_r);

    let rank = comptime!(space.rank());
    let (mr, nr) = comptime!((space.extent_at(rank - 2), space.extent_at(rank - 1) / vw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());

    // The reduce axes' extents come off the operands' merged space, not the accumulator's: a
    // contracted axis is by definition absent from the accumulator, and an axis only one operand
    // spans still has to be walked.
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let reduce = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space).to_vec());
    let reduce_extents = comptime!(reduce.iter().map(|&a| merged.extent(a)).collect::<Vec<_>>());
    let kc = comptime!(reduce_extents.iter().product::<usize>());

    comptime!(assert_operand_shapes(
        &lhs.space, &rhs.space, &space, &reduce
    ));

    let lhs_view = lhs.nd::<IL, WPL, L>();
    let rhs_view = rhs.nd::<IR, WPR, V>();
    // Loop-invariant, and `comptime!`-bound so the `unroll` flag below stays a comptime binding:
    // `#[unroll(flag)]` silently rolls the loop when the macro cannot see `flag` as one.
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);

    // The fan-out walk names the lane with a comptime extract. Its final physical line can be
    // partial, just as the direct leaf's can, so retain a short tail rather than rejecting a
    // perfectly valid checked tile.
    let k_lines = comptime!(kc / lw);
    let k_tail = comptime!(kc % lw);

    let unroll_limit = comptime!(config.unroll_limit);
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

        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));

        // Unroll only when no mask, otherwise compilation too long.
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= unroll_limit && !lhs_check && !rhs_check && !acc_check);
        let mut c = block::seed(&mut acc, comptime!(mr), comptime!(nr), unroll);

        // One rhs line per accumulator column, reused by every row of the rank-1 update. Held
        // across the whole K walk rather than re-declared per step, so the trace allocates it once
        // however many lane bodies the fan-out below emits.
        let mut b = Array::<Vector<E, V>>::new(nr);

        if comptime!(lane_fanout && lw > 1) {
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
                );
            }
        }

        block::commit(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
    }
}

/// One gathered rank-1 update. `lane` names the component to take when the caller walks `K` as
/// (line, lane), so shader backends see a fixed `extract`; `None` is the flat walk, which resolves
/// the component from `p` instead.
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
) {
    let reduce_coords = unravel(&const_coords(reduce_extents), p.fcast::<u32>());

    #[unroll(unroll)]
    for n in 0..nr {
        let acc_coords = acc_cell_coords(batch, 0u32, n as u32);
        let pos = resolve_nd_coords(
            comptime!(rhs_space.clone()),
            comptime!(space.clone()),
            comptime!(reduce.clone()),
            &acc_coords,
            &reduce_coords,
            vw,
            false,
        );
        b[n] = Vector::<E, V>::cast_from(rhs_view.read(pos));
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let acc_coords = acc_cell_coords(batch, i as u32, 0u32);
        // `resolve_nd_coords` divides the fastest contracted coordinate by `lw` into a line index,
        // so this is the same position for every lane of one line.
        let pos = resolve_nd_coords(
            comptime!(lhs_space.clone()),
            comptime!(space.clone()),
            comptime!(reduce.clone()),
            &acc_coords,
            &reduce_coords,
            lw,
            false,
        );
        let lhs_line = lhs_view.read(pos);
        let a = if comptime!(lane.is_some()) {
            Vector::<E, V>::cast_from(lhs_line.extract(comptime!(lane.unwrap())))
        } else {
            Vector::<E, V>::cast_from(lhs_line.extract_dynamic(p % lw))
        };
        #[unroll(unroll)]
        for n in 0..nr {
            c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
        }
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
fn assert_operand_shapes(lhs: &Space, rhs: &Space, acc: &Space, reduce: &[Axis]) {
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
    assert!(
        rhs.axis_at(rhs.rank() - 1) == acc.axis_at(acc.rank() - 1),
        "contract gather: the rhs must line along the accumulator's innermost axis"
    );
}
