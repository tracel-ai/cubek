//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::*;

/// Fully unroll the `mr × nr` register block only up to this many cells; past it the
/// load/store loops run at runtime, since hundreds of inlined cells overflow the
/// optimizer's recursive block pass. An *edge-masked* block never fully unrolls
/// regardless of size: each guarded access is its own CFG branch (see [`mma_register`]).
const UNROLL_BLOCK: usize = 64;

/// Run the register microkernel over each batch matrix, reading operands through the
/// quant-transparent [`matrix_transparent`](Tile::matrix_transparent): a plain operand is a bare
/// matrix read, a quantized one dequantizes per read (no dequantize-into-`f32` fill). Either
/// operand may be the quantized one — the gemv weight is the RHS, an activation-times-weight the
/// LHS — so this dispatches each operand's storage element/packing (`0` plain, `1` native i8, `>1`
/// packed u32). Both quantized at once is not a real workload and is refused.
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
) {
    // `nr` is a line count (how many `Vector<V>` span `N`), so it divides `N` by the accumulator
    // width `V`. `mr` (rows) and `kc` (scalar `K`, off `rhs`) are unvectorized.
    let vw = rhs.vector_size();
    let lw = lhs.vector_size();
    let (mr, nr, kc) = comptime! {
        (
            space.extent_at(space.rank() - 2),
            space.extent_at(space.rank() - 1) / vw,
            rhs.space.extent_at(rhs.space.rank() - 2)
        )
    };
    let size!(V) = vw;
    let size!(L) = lw;

    // A plain operand serves `I =` its element, `WP =` its served width; native i8 serves `i8` at
    // the same width; packed serves `u32` at `width / pack` (the narrower physical line). `pack` is
    // the comptime dispatch key, mirroring `fill_from`'s storage-element choice.
    let lp = lhs.quant_pack();
    let rp = rhs.quant_pack();
    if comptime!(lp == 0 && rp == 0) {
        mma_register_typed::<E, EL, EL, L, L, ER, ER, V, V>(acc, lhs, rhs, space, mr, nr, kc, lw);
    } else if comptime!(rp == 0) {
        // Quantized LHS, plain RHS.
        if comptime!(lp == 1) {
            mma_register_typed::<E, EL, i8, L, L, ER, ER, V, V>(
                acc, lhs, rhs, space, mr, nr, kc, lw,
            );
        } else {
            let size!(WPL) = comptime!(lw / lp);
            mma_register_typed::<E, EL, u32, WPL, L, ER, ER, V, V>(
                acc, lhs, rhs, space, mr, nr, kc, lw,
            );
        }
    } else if comptime!(lp == 0) {
        // Plain LHS, quantized RHS — the gemv weight.
        if comptime!(rp == 1) {
            mma_register_typed::<E, EL, EL, L, L, ER, i8, V, V>(
                acc, lhs, rhs, space, mr, nr, kc, lw,
            );
        } else {
            let size!(WPR) = comptime!(vw / rp);
            mma_register_typed::<E, EL, EL, L, L, ER, u32, WPR, V>(
                acc, lhs, rhs, space, mr, nr, kc, lw,
            );
        }
    } else {
        comptime!(panic!(
            "register leaf: both operands quantized is not a supported direct-serve case"
        ));
    }
}

/// The batch-matrix loop for fixed lhs (`IL`/`WPL`) and rhs (`IR`/`WPR`) storage elements. Split
/// from the dispatch so the microkernel is written once regardless of which operand is quantized.
#[cube]
fn mma_register_typed<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    WPL: Size,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    WPR: Size,
    V: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] lw: usize,
) {
    let matrices = comptime! {
        let mut count = 1;
        for p in 0..space.rank() - 2 {
            count *= space.extent_at(p);
        }
        count
    };

    for j in 0..matrices {
        let l = lhs.matrix_transparent::<IL, WPL, L>(j);
        let r = rhs.matrix_transparent::<IR, WPR, V>(j);
        let mut a = acc.matrix_mut::<V>(j, comptime!(space.clone()));
        mma_register(&l, &r, &mut a, mr, nr, kc, lw);
    }
}

/// The microkernel. The `mr × nr` block of `V`-wide accumulators lives in registers: load once,
/// run `kc` rank-1 updates ([`outer_product`]), store once. `nr` counts `N`-lines. Operands arrive
/// as quant-transparent [`TileView`]s, so a quantized operand dequantizes per read.
#[cube]
fn mma_register<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    WPL: Size,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    WPR: Size,
    V: Size,
>(
    lhs: &TileView<'_, EL, IL, WPL, L, Coords2d>,
    rhs: &TileView<'_, ER, IR, WPR, V, Coords2d>,
    acc: &mut MatrixViewMut<'_, Vector<E, V>>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] l: usize,
) {
    // Unroll only when no mask, otherwise compilation too long
    let lhs_check = lhs.check();
    let rhs_check = rhs.check();
    let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs_check && !rhs_check && !acc.check);
    let mut c = Array::<Vector<E, V>>::new(mr * nr);

    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for j in 0..nr {
            c[i * nr + j] = acc.read((i as u32, j as u32));
        }
    }

    for p in 0..kc {
        outer_product::<E, EL, IL, WPL, L, ER, IR, WPR, V>(lhs, rhs, &mut c, p, mr, nr, unroll, l);
    }

    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for j in 0..nr {
            acc.write((i as u32, j as u32), c[i * nr + j]);
        }
    }
}

/// One rank-1 update at scalar depth `p`: `c += outer(A[:, p], B[p, :])`. `A[i, p]` is lane
/// `p % L` of `lhs`'s `(p / L)` `K`-line, broadcast and multiplied by `B`'s `V`-wide lines.
/// Each operand is read in its own element (`EL`/`ER`) and cast to the accumulate element `E`.
#[cube]
fn outer_product<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    WPL: Size,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    WPR: Size,
    V: Size,
>(
    lhs: &TileView<'_, EL, IL, WPL, L, Coords2d>,
    rhs: &TileView<'_, ER, IR, WPR, V, Coords2d>,
    c: &mut Array<Vector<E, V>>,
    p: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
    #[comptime] l: usize,
) {
    // `p` is a runtime K step (the `kc` loop never unrolls), so the line index and lane
    // fold are runtime too; `extract` takes a runtime index. `unroll` mirrors the caller's
    // masked-block decision so a masked tile keeps these inner loops at runtime too. `l` is the
    // `lhs` line width (passed in: a reconstructed `DynamicSize` can't answer `L::value()` here).
    let mut b = Array::<Vector<E, V>>::new(nr);
    #[unroll(unroll)]
    for j in 0..nr {
        // Reads past the operand's logical bound contribute 0 to the contraction; `rhs` widens
        // from `ER` into the accumulate element `E`.
        b[j] = Vector::<E, V>::cast_from(rhs.read((p as u32, j as u32)));
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let lhs_line = lhs.read((i as u32, (p / l) as u32));
        let scalar = lhs_line.extract(p % l);
        // Broadcast the `EL` lane across the `V` line and widen into `E` in one cast.
        let a = Vector::<E, V>::cast_from(scalar);
        #[unroll(unroll)]
        for j in 0..nr {
            // Explicit fused multiply-add: `+= a * b` lowers to a separate mul + dependent add
            // (no fast-math contraction on the CPU backend), which doubles the FP instruction
            // count and serializes the accumulate. `fma` emits one fused op (`fmla`).
            c[i * nr + j] = fma(a, b[j], c[i * nr + j]);
        }
    }
}
