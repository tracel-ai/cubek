//! N-D register microkernel for operations with multiple contracted axes or projected operands.

use cubecl::prelude::*;

use super::base::{load_accumulators, store_accumulators};
use super::tuning::RegisterTuning;
use crate::*;

/// N-D variant of [`mma_register_direct`](super::direct::mma_register_direct) for operations with
/// multiple contracted axes or projected operands.
#[cube]
pub(super) fn mma_register_gather<
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
    #[comptime] tuning: RegisterTuning,
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
        &lhs.space, &rhs.space, &space, &reduce, lw
    ));

    let lhs_view = lhs.nd::<IL, WPL, L>();
    let rhs_view = rhs.nd::<IR, WPR, V>();
    // Loop-invariant, and `comptime!`-bound so the `unroll` flag below stays a comptime binding:
    // `#[unroll(flag)]` silently rolls the loop when the macro cannot see `flag` as one.
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);

    // `K` walks the flattened reduce space as (line, lane): the lhs lines along the fastest
    // contracted axis and `assert_operand_shapes` has that axis's extent a multiple of `lw`, so
    // `lw` divides `kc` and the lane of step `line * lw + lane` is `lane` outright. Comptime, so
    // `extract` names a fixed component instead of the `frem` plus `extract_dynamic` a flat walk
    // needs, and the lhs line index is the same for every lane of one line.
    let k_lines = comptime!(kc / lw);

    let unroll_block = comptime!(tuning.unroll_block);

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
        let unroll = comptime!(mr * nr <= unroll_block && !lhs_check && !rhs_check && !acc_check);
        let mut c = load_accumulators(&mut acc, comptime!(mr), comptime!(nr), unroll);

        // One rhs line per accumulator column, reused by every row of the rank-1 update. Held
        // across the whole K walk rather than re-declared per step, so the trace allocates it once
        // however many lane bodies the fan-out below emits.
        let mut b = Array::<Vector<E, V>>::new(nr);

        for line in 0..k_lines {
            #[unroll]
            for lane in 0..lw {
                let reduce_coords = unravel(
                    &const_coords(comptime!(reduce_extents.clone())),
                    (line * lw + lane).fcast::<u32>(),
                );

                #[unroll(unroll)]
                for n in 0..nr {
                    let acc_coords = acc_cell_coords(&batch, 0u32, n as u32);
                    let pos = resolve_nd_coords(
                        comptime!(rhs.space.clone()),
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
                    let acc_coords = acc_cell_coords(&batch, i as u32, 0u32);
                    // `resolve_nd_coords` divides the fastest contracted coordinate by `lw` into a
                    // line index, so this is the same position for every lane of one line.
                    let pos = resolve_nd_coords(
                        comptime!(lhs.space.clone()),
                        comptime!(space.clone()),
                        comptime!(reduce.clone()),
                        &acc_coords,
                        &reduce_coords,
                        lw,
                        false,
                    );
                    let a = Vector::<E, V>::cast_from(lhs_view.read(pos).extract(lane));
                    #[unroll(unroll)]
                    for n in 0..nr {
                        c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                    }
                }
            }
        }

        store_accumulators(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
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
    lhs_vec_len: usize,
) {
    assert!(
        !reduce.is_empty(),
        "gather leaf: the operands contract no axis against the accumulator"
    );
    // `Space::contracted` merges lhs-first, so an axis only the rhs spans lands past every lhs
    // one and would take the `fastest` slot below. That reads as the lhs being lined wrong, which
    // it is not, so the real constraint is named here instead.
    for &axis in reduce {
        assert!(
            lhs.contains(axis),
            "gather leaf: the lhs must span every contracted axis, but {axis:?} is contracted by \
             the rhs alone"
        );
    }
    let fastest = reduce[reduce.len() - 1];
    assert!(
        lhs.axis_at(lhs.rank() - 1) == fastest,
        "gather leaf: the lhs must line along the fastest contracted axis {fastest:?}"
    );
    assert!(
        lhs_vec_len == 1 || lhs.extent(fastest).is_multiple_of(lhs_vec_len),
        "gather leaf: the lhs's line width {lhs_vec_len} must divide its fastest contracted axis's extent"
    );
    assert!(
        rhs.axis_at(rhs.rank() - 1) == acc.axis_at(acc.rank() - 1),
        "gather leaf: the rhs must line along the accumulator's innermost axis"
    );
}
