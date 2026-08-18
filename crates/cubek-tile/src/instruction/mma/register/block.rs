//! The K walk into an `mr × nr` register block, shared by both register-leaf callers.
//!
//! The block is a parameter, not something this owns, which is the only difference between the
//! two: the memory-backed leaf ([`mma_register_direct`](super::direct::mma_register_direct))
//! seeds a local one from the accumulator and commits it back per visit, while a promoted
//! [`RegisterData`](crate::RegisterData) *is* the accumulator and keeps it across the whole walk.
//! How the block is contracted is the same either way.

use cubecl::prelude::*;

use crate::*;

/// `c += lhs · rhs` over the block: `kc` rank-1 updates into the `mr × nr` lines of `c`.
///
/// `K` is walked as (line, lane) rather than as one flat scalar step. The lhs lines along `K`, so a
/// flat walk has to divide and remainder by `lw` every step and can only reach the element through
/// `extract_dynamic`, which the backends lower through memory. Splitting the walk makes the lane a
/// comptime index, so `extract` folds to a fixed component, and the `lw` reads of one K-line become
/// loop-invariant across the lane fan-out.
///
/// `kc` arrives whole rather than pre-split so the line count and the tail are derived in one place
/// for both callers.
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
) {
    let k_lines = comptime!(kc / lw);
    let k_tail = comptime!(kc % lw);

    // One rhs line per accumulator column, reused by every row of the rank-1 update. Held across
    // the whole K walk rather than re-declared per step, so the trace allocates it once however
    // many lane bodies the fan-out below emits.
    let mut b = Array::<Vector<E, V>>::new(nr);

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
