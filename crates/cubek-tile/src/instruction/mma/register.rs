//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

/// Fully unroll the `mr × nr` register block only up to this many cells; past it the
/// load/store loops run at runtime, since hundreds of inlined cells overflow the
/// optimizer's recursive block pass. An *edge-masked* block never fully unrolls
/// regardless of size: each guarded access is its own CFG branch (see [`mma_register_direct`]).
const UNROLL_BLOCK: usize = 64;

/// Run the register microkernel over each batch matrix, reading operands through the
/// quant-transparent [`matrix_transparent`](Tile::matrix_transparent): a plain operand is a bare
/// matrix read, a quantized one dequantizes per read (no dequantize-into-`f32` fill). Either
/// operand may be the quantized one (the gemv weight is the RHS, an activation-times-weight the
/// LHS), so this dispatches each operand's storage element/packing (`0` plain, `1` native i8, `>1`
/// packed u32). Both quantized at once is not a real workload and is refused.
///
/// The 2-D microkernel reads each operand as a batch matrix, which is only a description of it
/// when one axis is contracted *and* a logical coordinate is a physical one. Either condition
/// failing takes the N-D nest ([`mma_register_gather`]); they are independent, so a stencil
/// contracting a single axis is a gather just as much as a two-axis reduce is.
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
) {
    let size!(L) = lhs.vector_size();
    let size!(V) = rhs.vector_size();

    // Each operand's storage element is `i8` native / `u32` packed / the served element when plain;
    // its pack factor narrows the physical line, derived in the microkernel. `quant_pack` is `0`
    // plain / `1` native / `>1` the packed-u32 factor. At most one operand is quantized (the gemv
    // weight is the RHS, an activation·weight the LHS); both at once is refused.
    let pack_l = lhs.quant_pack();
    let pack_r = rhs.quant_pack();
    comptime!(assert!(
        pack_l == 0 || pack_r == 0,
        "register leaf: both operands quantized is not a supported direct-serve case"
    ));

    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let nd = comptime!(
        Space::contracted(&[&lhs.space, &rhs.space], &space).len() > 1
            || lhs_gathered
            || rhs_gathered
    );
    if nd {
        if comptime!(pack_l == 1) {
            mma_register_gather::<E, EL, i8, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
        } else if comptime!(pack_l > 1) {
            mma_register_gather::<E, EL, u32, L, ER, ER, V>(acc, lhs, rhs, space, pack_l, 1usize);
        } else if comptime!(pack_r == 1) {
            mma_register_gather::<E, EL, EL, L, ER, i8, V>(acc, lhs, rhs, space, 1usize, 1usize);
        } else if comptime!(pack_r > 1) {
            mma_register_gather::<E, EL, EL, L, ER, u32, V>(acc, lhs, rhs, space, 1usize, pack_r);
        } else {
            mma_register_gather::<E, EL, EL, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
        }
    } else if comptime!(pack_l == 1) {
        mma_register_direct::<E, EL, i8, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
    } else if comptime!(pack_l > 1) {
        mma_register_direct::<E, EL, u32, L, ER, ER, V>(acc, lhs, rhs, space, pack_l, 1usize);
    } else if comptime!(pack_r == 1) {
        mma_register_direct::<E, EL, EL, L, ER, i8, V>(acc, lhs, rhs, space, 1usize, 1usize);
    } else if comptime!(pack_r > 1) {
        mma_register_direct::<E, EL, EL, L, ER, u32, V>(acc, lhs, rhs, space, 1usize, pack_r);
    } else {
        mma_register_direct::<E, EL, EL, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
    }
}

/// The register microkernel for a fixed lhs (`IL`) and rhs (`IR`) storage element: over each batch
/// matrix, the `mr × nr` block of `V`-wide accumulators lives in registers (load once, `kc` rank-1
/// updates, store once). `pack_l`/`pack_r` narrow each operand's physical line (`served / pack`,
/// `1` for plain/native). The storage element per operand is the price of a typed quant view
/// (`#[cube]` takes no `impl Trait`, so the view can't be erased behind a `read` trait); `#[cube]`
/// inlines at trace time, so folding the rank-1 step in here costs nothing over a separate fn.
///
/// The 2-D form its reads assume: `mat` indexes a batch matrix, `(row, k)` and `(k, col)` address
/// the operands. [`mma_register_memory`] routes anything else to [`mma_register_gather`], so the
/// two conditions below are re-asserted rather than re-decided.
#[cube]
fn mma_register_direct<
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
) {
    comptime!(assert!(
        Space::contracted(&[&lhs.space, &rhs.space], &space).len() == 1,
        "register leaf: the 2-D microkernel contracts exactly one axis"
    ));
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered,
        "register leaf: a gathered operand has no 2-D matrix view; it needs the N-D nest"
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

    for mat in 0..matrices {
        let lhs = lhs.matrix_transparent::<IL, WPL, L>(mat);
        let rhs = rhs.matrix_transparent::<IR, WPR, V>(mat);
        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));

        // Unroll only when no mask, otherwise compilation too long.
        let lhs_check = comptime!(lhs.check);
        let rhs_check = comptime!(rhs.check);
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs_check && !rhs_check && !acc_check);
        let mut c = load_accumulators(&mut acc, comptime!(mr), comptime!(nr), unroll);

        // One rank-1 update per scalar `K` step `p`: `c += outer(A[:, p], B[p, :])`. `p` is a
        // runtime step (the `kc` loop never unrolls), so the line index and lane fold are runtime;
        // `extract` takes a runtime index. `A[i, p]` is lane `p % lw` of `lhs`'s `(p / lw)` K-line,
        // broadcast; `B`'s `V`-wide lines widen from `ER` into the accumulate element `E`. Reads
        // past the operands' logical bound contribute `0` to the contraction.
        for p in 0..kc {
            let mut b = Array::<Vector<E, V>>::new(nr);
            #[unroll(unroll)]
            for n in 0..nr {
                b[n] = Vector::<E, V>::cast_from(rhs.read((p as u32, n as u32)));
            }
            #[unroll(unroll)]
            for i in 0..mr {
                let lhs_line = lhs.read((i as u32, (p / lw) as u32));
                let a = Vector::<E, V>::cast_from(lhs_line.extract_dynamic(p % lw));
                #[unroll(unroll)]
                for n in 0..nr {
                    // Explicit `fma`: `+= a * b` lowers to a separate mul + dependent add (no
                    // fast-math contraction on the CPU backend), doubling the FP instruction count
                    // and serializing the accumulate. `fma` emits one fused op (`fmla`).
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }

        store_accumulators(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
    }
}

/// N-D variant of [`mma_register_direct`] for operations with multiple contracted axes
/// or projected operands.
#[cube]
fn mma_register_gather<
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
        let lhs_check = comptime!(lhs_view.check);
        let rhs_check = comptime!(rhs_view.check);
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs_check && !rhs_check && !acc_check);
        let mut c = load_accumulators(&mut acc, comptime!(mr), comptime!(nr), unroll);

        for p in 0..kc {
            let reduce_coords = unravel(
                &const_coords(comptime!(reduce_extents.clone())),
                p.fcast::<u32>(),
            );
            // `lhs` lines along the fastest contracted axis (`assert_operand_shapes`), so that
            // axis's coordinate splits into the line index `resolve_nd_coords` divides out and the
            // lane within it, broadcast below.
            let lane = reduce_coords
                .at(comptime!(reduce.len() - 1))
                .frem(comptime!(lw as u32))
                .fcast::<usize>();

            let mut b = Array::<Vector<E, V>>::new(nr);
            #[unroll(unroll)]
            for n in 0..nr {
                let pos = resolve_nd_coords(
                    comptime!(rhs.space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    &batch,
                    &reduce_coords,
                    0u32,
                    n as u32,
                    vw,
                );
                b[n] = Vector::<E, V>::cast_from(rhs_view.read(pos));
            }
            #[unroll(unroll)]
            for i in 0..mr {
                let pos = resolve_nd_coords(
                    comptime!(lhs.space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    &batch,
                    &reduce_coords,
                    i as u32,
                    0u32,
                    lw,
                );
                let a = Vector::<E, V>::cast_from(lhs_view.read(pos).extract_dynamic(lane));
                #[unroll(unroll)]
                for n in 0..nr {
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }

        store_accumulators(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
    }
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the rank-1
/// updates never touch memory. Shared by both microkernels: how a cell is addressed differs, how
/// the block is held does not. `unroll` is the caller's decision, not a size test here, because a
/// masked block must stay rolled whatever its size ([`UNROLL_BLOCK`]).
#[cube]
fn load_accumulators<E: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            c[i * nr + n] = acc.seed((i as u32, n as u32));
        }
    }
    c
}

/// The twin of [`load_accumulators`]: commit the block back once the whole reduce is folded into
/// it. Through [`AccumulateView`], so a lane-split accumulator reduces across lanes on the way out
/// rather than the leaf knowing it was split.
#[cube]
fn store_accumulators<E: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    c: Array<Vector<E, V>>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) {
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            acc.commit((i as u32, n as u32), c[i * nr + n]);
        }
    }
}

/// The coordinate `operand` is read at, one entry per axis of its own space, assembled from the
/// three sources a step has: the accumulator cell (`row`/`col`), the reduce nest, and the batch.
/// Every operand axis falls in exactly one of them, since an axis absent from the accumulator is
/// contracted by definition.
///
/// `width` is the operand's line width, and only its innermost axis is addressed in lines, so the
/// division applies there alone. The `col` branch needs none: it is already a line index, the
/// accumulator's innermost axis being the one the rhs lines along ([`assert_operand_shapes`]).
/// The reduce branch does: the lhs lines along the fastest contracted axis, whose coordinate is a
/// scalar step, and the lane within the line is folded back by the caller.
#[cube]
fn resolve_nd_coords(
    #[comptime] operand: Space,
    #[comptime] acc: Space,
    #[comptime] reduce: Vec<Axis>,
    batch: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] width: usize,
) -> CoordsDyn {
    let operand_rank = comptime!(operand.rank());
    let acc_rank = comptime!(acc.rank());
    let mut out = CoordsDyn::new();

    #[unroll]
    for axis_idx in 0..operand_rank {
        let axis = comptime!(operand.axis_at(axis_idx));
        let axis_coord = if comptime!(axis == acc.axis_at(acc_rank - 2)) {
            row
        } else if comptime!(axis == acc.axis_at(acc_rank - 1)) {
            col
        } else if comptime!(reduce.contains(&axis)) {
            let reduce_coord =
                reduce_coords.at(comptime!(reduce.iter().position(|&r| r == axis).unwrap()));
            if comptime!(axis_idx == operand_rank - 1 && width > 1) {
                reduce_coord.fdiv(comptime!(width as u32))
            } else {
                reduce_coord
            }
        } else {
            batch.at(comptime!(acc.position(axis)))
        };
        out.push(axis_coord);
    }

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
