//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.

use cubecl::prelude::*;

use crate::*;

/// Fully unroll the `mr × nr` register block only up to this many cells; past it the
/// load/store loops run at runtime, since hundreds of inlined cells overflow the
/// optimizer's recursive block pass. An *edge-masked* block never fully unrolls
/// regardless of size: each guarded access is its own CFG branch (see [`mma_register`]).
const UNROLL_BLOCK: usize = 64;

/// Run the register microkernel over each batch matrix. `mr × nr` are the accumulator's
/// trailing axes (`nr` in `N`-lines); `kc` is scalar `K`, read off `rhs` (whose `K` is unlined).
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
    let (mr, nr, kc) = comptime! {
        (
            space.extent_at(space.rank() - 2),
            space.extent_at(space.rank() - 1) / vw,
            rhs.space.extent_at(rhs.space.rank() - 2)
        )
    };

    let matrices = comptime! {
        let mut count = 1;
        for p in 0..space.rank() - 2 {
            count *= space.extent_at(p);
        }
        count
    };

    let lw = lhs.vector_size();
    let size!(V) = vw;
    let size!(L) = lw;

    for j in 0..matrices {
        let l = lhs.matrix::<L>(j);
        let r = rhs.matrix::<V>(j);
        let mut a = acc.matrix_mut::<V>(j, comptime!(space.clone()));
        mma_register(&l, &r, &mut a, mr, nr, kc, lw);
    }
}

/// The microkernel. The `mr × nr` block of `V`-wide accumulators lives in registers: load once,
/// run `kc` rank-1 updates ([`outer_product`]), store once. `nr` counts `N`-lines.
#[cube]
fn mma_register<E: Numeric, EL: Numeric, ER: Numeric, L: Size, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    acc: &mut MatrixViewMut<'_, Vector<E, V>>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] l: usize,
) {
    // Unroll only when no mask, otherwise compilation too long
    let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs.check && !rhs.check && !acc.check);
    let mut c = Array::<Vector<E, V>>::new(mr * nr);

    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for j in 0..nr {
            c[i * nr + j] = acc.read((i as u32, j as u32));
        }
    }

    for p in 0..kc {
        outer_product::<E, EL, ER, L, V>(lhs, rhs, &mut c, p, mr, nr, unroll, l);
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
fn outer_product<E: Numeric, EL: Numeric, ER: Numeric, L: Size, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
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
