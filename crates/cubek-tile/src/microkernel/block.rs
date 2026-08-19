//! The `mr × nr` register block: seed it from the accumulator, contract into it, commit it back.
//!
//! The block is a parameter, not something this owns, which is the only difference between its
//! two callers: the memory-backed nest ([`contract_direct`](super::contract)) seeds a local one
//! and commits it back per visit, while a promoted [`RegisterData`] *is* the
//! accumulator and keeps it across the whole walk. How the block is contracted is the same either
//! way.

use cubecl::prelude::*;

use crate::*;

/// `c += lhs · rhs` over the block: `kc` rank-1 updates into the `mr × nr` lines of `c`.
///
/// If `lane_fanout` is true (GPU), `K` is walked as (line, lane) with fixed comptime extracts,
/// avoiding dynamic vector extraction on shader backends. If false (CPU or scalar lines), `K`
/// is walked as a flat scalar loop which LLVM vectorizes more efficiently without loop body bloat.
#[cube]
pub(crate) fn contract_block<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    c: &mut Array<Vector<E, V>>,
    #[comptime] lw: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
) {
    let mut b = Array::<Vector<E, V>>::new(nr);

    if comptime!(lane_fanout && lw > 1) {
        let k_lines = comptime!(kc / lw);
        let k_tail = comptime!(kc % lw);

        for line in 0..k_lines {
            #[unroll]
            for lane in 0..lw {
                rank1_update::<E, EL, L, ER, V>(
                    lhs,
                    rhs,
                    c,
                    &mut b,
                    (line * lw + lane) as u32,
                    line as u32,
                    lane,
                    mr,
                    nr,
                    unroll,
                );
            }
        }

        // A line width that does not divide `kc` leaves a partial last line. Its lane count is comptime
        // too, so the tail is straight-line code rather than a second, dynamic walk.
        #[unroll]
        for lane in 0..k_tail {
            rank1_update::<E, EL, L, ER, V>(
                lhs,
                rhs,
                c,
                &mut b,
                comptime!(k_lines * lw + lane) as u32,
                comptime!(k_lines) as u32,
                lane,
                mr,
                nr,
                unroll,
            );
        }
    } else {
        // Flat scalar walk (CPU or scalar lines)
        for p in 0..kc {
            #[unroll(unroll)]
            for n in 0..nr {
                b[n] = Vector::<E, V>::cast_from(rhs.read((p as u32, n as u32)));
            }
            #[unroll(unroll)]
            for i in 0..mr {
                let lhs_line = lhs.read((i as u32, (p / lw) as u32));
                let a = if comptime!(lw == 1) {
                    Vector::<E, V>::cast_from(lhs_line.extract(0usize))
                } else {
                    Vector::<E, V>::cast_from(lhs_line.extract_dynamic(p % lw))
                };
                #[unroll(unroll)]
                for n in 0..nr {
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }
    }
}

/// One rank-1 update `c += outer(A[:, k], B[k, :])` at scalar contraction step `k`, taking the lhs
/// from element `lane` of its `k_line`-th K-line. `B`'s `V`-wide lines widen from `ER` into the
/// accumulate element `E`; reads past the operands' logical bound contribute `0` to the
/// contraction.
///
/// `lane` is comptime, which is the whole point of walking `K` as (line, lane): `extract` names a
/// fixed component, and the `mr` line reads are the same reads for every lane of one line, so the
/// backend folds the fan-out's copies into one.
#[cube]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    k: u32,
    k_line: u32,
    #[comptime] lane: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) {
    #[unroll(unroll)]
    for n in 0..nr {
        b[n] = Vector::<E, V>::cast_from(rhs.read((k, n as u32)));
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let a = Vector::<E, V>::cast_from(lhs.read((i as u32, k_line)).extract(lane));
        #[unroll(unroll)]
        for n in 0..nr {
            // Explicit `fma`: `+= a * b` lowers to a separate mul + dependent add (no fast-math
            // contraction on the CPU backend), doubling the FP instruction count and serializing
            // the accumulate. `fma` emits one fused op (`fmla`).
            c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
        }
    }
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the rank-1
/// updates never touch memory. Shared by both nests: how a cell is addressed differs, how the
/// block is held does not. `unroll` is the caller's decision, not a size test here, because a
/// masked block must stay rolled whatever its size.
#[cube]
pub(crate) fn load_accumulators<E: Numeric, V: Size>(
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
            c[i * nr + n] = acc.seed((i as u32, n as u32), comptime!(LeafOp::Sum));
        }
    }
    c
}

/// The twin of [`load_accumulators`]: commit the block back once the whole reduce is folded into
/// it. Through [`AccumulateView`], so a lane-split accumulator reduces across lanes on the way out
/// rather than the leaf knowing it was split.
#[cube]
pub(crate) fn store_accumulators<E: Numeric, V: Size>(
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
            acc.commit((i as u32, n as u32), c[i * nr + n], comptime!(LeafOp::Sum));
        }
    }
}
