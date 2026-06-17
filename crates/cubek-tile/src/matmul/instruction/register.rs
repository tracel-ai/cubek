//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.

use cubecl::prelude::*;

use crate::*;

/// Fully unroll the `mr × nr` register block only up to this many cells. Past it the
/// load/store loops run at runtime: a larger block (the heuristic sizes tiles for L1, not
/// registers) would inline hundreds of cells into one straight chain, overflowing the
/// optimizer's recursive block pass. An *edge-masked* block never fully unrolls regardless
/// of size — each guarded load/store is its own CFG branch, so `mr × nr` of them in a chain
/// blow the recursive passes even well under this cap (see [`mma_register`]).
const UNROLL_BLOCK: usize = 64;

/// Run the register microkernel over each batch matrix. `mr × nr` are the accumulator's
/// trailing axes (`nr` in `N`-lines); `kc` is scalar `K`, read off `rhs` (whose `K` is unlined).
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, EL: Numeric, ER: Numeric, L: Size, V: Size>(
    acc: &mut MemData<Vector<E, V>>,
    lhs: &Tile<Vector<EL, L>>,
    rhs: &Tile<Vector<ER, V>>,
    #[comptime] space: Space,
) {
    let (mr, nr, kc) = comptime! {
        (
            space.extent_at(space.rank() - 2),
            space.extent_at(space.rank() - 1),
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

    for j in 0..matrices {
        let l = lhs.matrix(j);
        let r = rhs.matrix(j);
        let mut a = acc.matrix_mut(j, comptime!(space.clone()));
        mma_register::<E, EL, ER, L, V>(&l, &r, &mut a, mr, nr, kc);
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
        outer_product::<E, EL, ER, L, V>(lhs, rhs, &mut c, p, mr, nr, unroll);
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
) {
    // `p` is a runtime K step (the `kc` loop never unrolls), so the line index and lane
    // fold are runtime too; `extract` takes a runtime index. `unroll` mirrors the caller's
    // masked-block decision so a masked tile keeps these inner loops at runtime too.
    let l = comptime!(L::value());
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
            c[i * nr + j] += a * b[j];
        }
    }
}
