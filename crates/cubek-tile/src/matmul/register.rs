//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.

use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::Coords2d},
};

use crate::*;

/// Fully unroll the `mr × nr` register block only up to this many cells. Past it the
/// load/store loops run at runtime: a larger block (the heuristic sizes tiles for L1, not
/// registers) would inline hundreds of cells — and, when edge-masked, as many bounds
/// branches — into one straight chain, overflowing the optimizer's recursive block pass.
const UNROLL_BLOCK: usize = 64;

/// Run the register microkernel over each batch matrix. `mr × nr` are the accumulator's
/// trailing axes (`nr` in `N`-lines); `kc` is scalar `K`, read off `rhs` (whose `K` is unlined).
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, L: Size, V: Size>(
    acc: &mut MemData<Vector<E, V>>,
    lhs: &Tile<Vector<E, L>>,
    rhs: &Tile<Vector<E, V>>,
    #[comptime] space: Space,
    #[comptime] acc_check: bool,
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

    // Edge masking: `acc` (the output) guards its stores, `lhs`/`rhs` zero their reads.
    // In the staged schedule the inputs are smem (never checked); in the direct schedule
    // they are gmem and inherit their operand's flag. `acc_check` is the output tile's
    // flag, passed in since only the payload (not the `Tile`) reaches here.
    let lhs_check = comptime!(lhs.check);
    let rhs_check = comptime!(rhs.check);

    for j in 0..matrices {
        let l = lhs.matrix(j);
        let r = rhs.matrix(j);
        let mut a = acc.matrix_mut(j, comptime!(space.clone()));
        mma_register::<E, L, V>(&l, &r, &mut a, mr, nr, kc, acc_check, lhs_check, rhs_check);
    }
}

/// The microkernel. The `mr × nr` block of `V`-wide accumulators lives in registers: load once,
/// run `kc` rank-1 updates ([`outer_product`]), store once. `nr` counts `N`-lines.
#[cube]
fn mma_register<E: Numeric, L: Size, V: Size>(
    lhs: &View<'_, Vector<E, L>, Coords2d>,
    rhs: &View<'_, Vector<E, V>, Coords2d>,
    acc: &mut ViewMut<'_, Vector<E, V>, Coords2d>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] acc_check: bool,
    #[comptime] lhs_check: bool,
    #[comptime] rhs_check: bool,
) {
    let unroll = comptime!(mr * nr <= UNROLL_BLOCK);
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for j in 0..nr {
            // An out-of-bounds accumulator cell reads 0; its store is skipped below, so
            // the overhang never round-trips through gmem.
            c[i * nr + j] = if comptime!(acc_check) {
                acc.read_checked((i as u32, j as u32))
            } else {
                acc.read((i as u32, j as u32))
            };
        }
    }

    for p in 0..kc {
        outer_product::<E, L, V>(lhs, rhs, &mut c, p, mr, nr, lhs_check, rhs_check);
    }

    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for j in 0..nr {
            if comptime!(acc_check) {
                acc.write_checked((i as u32, j as u32), c[i * nr + j]);
            } else {
                acc.write((i as u32, j as u32), c[i * nr + j]);
            }
        }
    }
}

/// One rank-1 update at scalar depth `p`: `c += outer(A[:, p], B[p, :])`. `A[i, p]` is lane
/// `p % L` of `lhs`'s `(p / L)` `K`-line, broadcast and multiplied by `B`'s `V`-wide lines.
#[cube]
fn outer_product<E: Numeric, L: Size, V: Size>(
    lhs: &View<'_, Vector<E, L>, Coords2d>,
    rhs: &View<'_, Vector<E, V>, Coords2d>,
    c: &mut Array<Vector<E, V>>,
    p: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] lhs_check: bool,
    #[comptime] rhs_check: bool,
) {
    // `p` is a runtime K step (the `kc` loop never unrolls), so the line index and lane
    // fold are runtime too; `extract` takes a runtime index.
    let unroll = comptime!(mr * nr <= UNROLL_BLOCK);
    let l = comptime!(L::value());
    let mut b = Array::<Vector<E, V>>::new(nr);
    #[unroll(unroll)]
    for j in 0..nr {
        // Reads past the operand's logical bound contribute 0 to the contraction.
        b[j] = if comptime!(rhs_check) {
            rhs.read_checked((p as u32, j as u32))
        } else {
            rhs.read((p as u32, j as u32))
        };
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let lhs_line = if comptime!(lhs_check) {
            lhs.read_checked((i as u32, (p / l) as u32))
        } else {
            lhs.read((i as u32, (p / l) as u32))
        };
        let scalar = lhs_line.extract(p % l);
        let a = Vector::<E, V>::cast_from(scalar);
        #[unroll(unroll)]
        for j in 0..nr {
            c[i * nr + j] += a * b[j];
        }
    }
}
