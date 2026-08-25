//! The `mr × nr` register block: seed it from the accumulator, contract into it, commit it back.
//!
//! The block is a parameter, not something this owns, which is the only difference between its
//! two callers: the memory-backed nest ([`contract`](super::contract)) seeds a local one
//! and commits it back per visit, while a promoted [`RegisterData`] *is* the
//! accumulator and keeps it across the whole walk. How the block is contracted is the same either
//! way.

use cubecl::prelude::*;

use crate::instruction::registers::horizontal;
use crate::*;

/// `c += lhs · rhs` over the block: `kc / served` steps into the `mr × nr` lines of `c`.
///
/// A step consumes `served` contracted values ([`Space::served`]). Past one, both operands line
/// along the contracted axis and the block's lanes are partials of one cell that [`commit`] folds.
/// At one, the rhs lines along the accumulator and the lhs's line is taken a lane at a time: as
/// (line, lane) with fixed comptime extracts when `lane_fanout` (GPU), else as a flat scalar loop.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    c: &mut Array<Vector<E, V>>,
    #[comptime] lw: usize,
    #[comptime] served: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
) {
    let mut b = Array::<Vector<E, V>>::new(nr);

    if comptime!(served > 1) {
        for line in 0..comptime!(kc / served) {
            rank1_update::<E, EL, L, ER, V>(
                lhs,
                rhs,
                c,
                &mut b,
                0usize,
                line as u32,
                comptime!(None),
                served,
                lw,
                mr,
                nr,
                unroll,
            );
        }
    } else if comptime!(lane_fanout && lw > 1) {
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
                    line * lw + lane,
                    line as u32,
                    comptime!(Some(lane)),
                    served,
                    lw,
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
                comptime!(k_lines * lw + lane),
                comptime!(k_lines) as u32,
                comptime!(Some(lane)),
                served,
                lw,
                mr,
                nr,
                unroll,
            );
        }
    } else {
        // Flat scalar walk (CPU or scalar lines)
        for p in 0..kc {
            rank1_update::<E, EL, L, ER, V>(
                lhs,
                rhs,
                c,
                &mut b,
                p,
                (p / lw) as u32,
                comptime!(None),
                served,
                lw,
                mr,
                nr,
                unroll,
            );
        }
    }
}

/// One step `c += outer(A[:, k], B[k, :])`, at scalar contraction step `k` off the `k_line`-th
/// K-line of each lhs row. Reads past the operands' logical bound contribute `0`.
///
/// At `served > 1` both reads are whole lines off the contracted axis, which is why the rhs is
/// addressed `(n, k_line)` there and `(k, n)` otherwise.
///
/// `lane` names the component to take when the caller walks `K` as (line, lane), so `extract`
/// names a fixed component and the backend folds the fan-out's `mr` repeated line reads into one.
/// `None` is the flat walk, which resolves the component from `k`. `k_line` stays a parameter so
/// each lane body sees one loop-invariant line index.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    k: usize,
    k_line: u32,
    #[comptime] lane: Option<usize>,
    #[comptime] served: usize,
    #[comptime] lw: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) {
    #[unroll(unroll)]
    for n in 0..nr {
        if comptime!(served > 1) {
            b[n] = Vector::<E, V>::cast_from(rhs.read((n as u32, k_line)));
        } else {
            b[n] = Vector::<E, V>::cast_from(rhs.read((k as u32, n as u32)));
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let lhs_line = lhs.read((i as u32, k_line));
        let a = if comptime!(served > 1) {
            Vector::<E, V>::cast_from(lhs_line)
        } else if comptime!(lane.is_some()) {
            Vector::<E, V>::cast_from(lhs_line.extract(comptime!(lane.unwrap())))
        } else if comptime!(lw == 1) {
            Vector::<E, V>::cast_from(lhs_line.extract(0usize))
        } else {
            Vector::<E, V>::cast_from(lhs_line.extract_dynamic(k % lw))
        };
        #[unroll(unroll)]
        for n in 0..nr {
            // One step of the sum-product semiring, which is a single `fma`: `+= a * b` would
            // lower to a separate mul + dependent add (no fast-math contraction on the CPU
            // backend), doubling the FP instruction count and serializing the accumulate.
            c[i * nr + n] = Semiring::SUM_PROD.step::<Vector<E, V>>(a, b[n], c[i * nr + n]);
        }
    }
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the steps
/// never touch memory. The algebra is the view's, stated where it was built.
///
/// At `served > 1` the block's lanes are partials of one cell, so the accumulator's value seeds
/// lane 0 alone and the rest start at the identity.
#[cube]
pub(crate) fn seed<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    #[comptime] served: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            let cell = acc.seed((i as u32, n as u32));
            if comptime!(served > 1) {
                let mut lanes = Vector::<E, V>::cast_from(Monoid::identity::<E>(Monoid::Sum));
                lanes.insert(0usize, cell.extract(0usize));
                c[i * nr + n] = lanes;
            } else {
                c[i * nr + n] = Vector::<E, V>::cast_from(cell);
            }
        }
    }
    c
}

/// The twin of [`seed`]: commit the block back once the whole contraction is folded into it,
/// collapsing the block's lanes first where they hold partials of one cell (`served > 1`).
/// Through [`AccumulateView`], so a lane-split accumulator reduces across lanes on the way out
/// rather than the leaf knowing it was split.
#[cube]
pub(crate) fn commit<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    c: Array<Vector<E, V>>,
    #[comptime] served: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) {
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            let cell = c[i * nr + n];
            if comptime!(served > 1) {
                let total = horizontal::vector::<E, V>(cell, served, Monoid::Sum);
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(total));
            } else {
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(cell));
            }
        }
    }
}
