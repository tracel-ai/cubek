//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.

use cubecl::prelude::*;

use crate::*;

/// Fully unroll the `mr × nr` register block only up to this many cells; past it the
/// load/store loops run at runtime, since hundreds of inlined cells overflow the
/// optimizer's recursive block pass. An *edge-masked* block never fully unrolls
/// regardless of size: each guarded access is its own CFG branch (see [`mma_register_typed`]).
const UNROLL_BLOCK: usize = 64;

/// Run the register microkernel over each batch matrix, reading operands through the
/// quant-transparent [`matrix_transparent`](Tile::matrix_transparent): a plain operand is a bare
/// matrix read, a quantized one dequantizes per read (no dequantize-into-`f32` fill). Either
/// operand may be the quantized one (the gemv weight is the RHS, an activation-times-weight the
/// LHS), so this dispatches each operand's storage element/packing (`0` plain, `1` native i8, `>1`
/// packed u32). Both quantized at once is not a real workload and is refused.
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
    if comptime!(pack_l == 1) {
        mma_register_typed::<E, EL, i8, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
    } else if comptime!(pack_l > 1) {
        mma_register_typed::<E, EL, u32, L, ER, ER, V>(acc, lhs, rhs, space, pack_l, 1usize);
    } else if comptime!(pack_r == 1) {
        mma_register_typed::<E, EL, EL, L, ER, i8, V>(acc, lhs, rhs, space, 1usize, 1usize);
    } else if comptime!(pack_r > 1) {
        mma_register_typed::<E, EL, EL, L, ER, u32, V>(acc, lhs, rhs, space, 1usize, pack_r);
    } else {
        mma_register_typed::<E, EL, EL, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
    }
}

/// The register microkernel for a fixed lhs (`IL`) and rhs (`IR`) storage element: over each batch
/// matrix, the `mr × nr` block of `V`-wide accumulators lives in registers (load once, `kc` rank-1
/// updates, store once). `pack_l`/`pack_r` narrow each operand's physical line (`served / pack`,
/// `1` for plain/native). The storage element per operand is the price of a typed quant view
/// (`#[cube]` takes no `impl Trait`, so the view can't be erased behind a `read` trait); `#[cube]`
/// inlines at trace time, so folding the rank-1 step in here costs nothing over a separate fn.
#[cube]
fn mma_register_typed<
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
        let lhs_check = lhs.check();
        let rhs_check = rhs.check();
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs_check && !rhs_check && !acc_check);
        let mut c = Array::<Vector<E, V>>::new(mr * nr);

        #[unroll(unroll)]
        for i in 0..mr {
            #[unroll(unroll)]
            for n in 0..nr {
                c[i * nr + n] = acc.seed((i as u32, n as u32));
            }
        }

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
                let a = Vector::<E, V>::cast_from(lhs_line.extract(p % lw));
                #[unroll(unroll)]
                for n in 0..nr {
                    // Explicit `fma`: `+= a * b` lowers to a separate mul + dependent add (no
                    // fast-math contraction on the CPU backend), doubling the FP instruction count
                    // and serializing the accumulate. `fma` emits one fused op (`fmla`).
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }

        #[unroll(unroll)]
        for i in 0..mr {
            #[unroll(unroll)]
            for n in 0..nr {
                acc.commit((i as u32, n as u32), c[i * nr + n]);
            }
        }
    }
}
