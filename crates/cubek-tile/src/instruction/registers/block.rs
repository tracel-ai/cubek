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
    #[comptime] semiring: Semiring,
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
                semiring,
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
                    semiring,
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
                semiring,
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
                semiring,
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
    #[comptime] semiring: Semiring,
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
            // One step of the accumulation's own semiring, a single `fma` where that is the
            // ordinary one: `+= a * b` would lower to a separate mul + dependent add (no
            // fast-math contraction on the CPU backend), doubling the FP instruction count and
            // serializing the accumulate.
            c[i * nr + n] = semiring.step::<Vector<E, V>>(a, b[n], c[i * nr + n]);
        }
    }
}

/// What [`seed`] and [`commit`] both need to hold before they spread a block column's lanes across
/// several sink cells: the lanes mean one thing at a time, and a spread one addresses cells the
/// accumulator serves singly.
fn assert_spread(served: usize, spread: usize, accumulator_width: usize, who: &str) {
    assert!(
        served == 1 || spread == 1,
        "{who}: lanes cannot hold contracted partials (served {served}) and neighbouring \
         sink cells (spread {spread}) at once"
    );
    assert!(
        spread == 1 || accumulator_width == 1,
        "{who}: a spread block scatters one lane per sink cell, so the accumulator must be \
         served scalar (it is {accumulator_width} wide)"
    );
}

/// Whether a spread block's lanes have to be tested against the sink's extent before they touch
/// it. The N-D nest rounds `nr` up, so the last column's spare lanes address cells past `cols`
/// exactly when `spread` does not divide it. Nothing masks them downstream: an unchecked
/// [`AccumulateView`] writes straight through.
fn spread_guard(spread: usize, cols: usize) -> bool {
    spread > 1 && !cols.is_multiple_of(spread)
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the steps
/// never touch memory. The algebra is the view's, stated where it was built.
///
/// At `served > 1` the block's lanes are partials of one cell, so the accumulator's value seeds
/// lane 0 alone and the rest start at the identity.
///
/// At `spread > 1` they instead hold neighbouring cells of a scalar sink. A padded shared-memory
/// operand serves whole lines even when its global source and sink are scalar, so each block
/// column gathers `spread` sink cells into its lanes. `cols` is the sink's own innermost extent,
/// which the block's `nr * spread` lanes overhang when `spread` does not divide it
/// ([`spread_guard`]).
#[cube]
pub(crate) fn seed<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    #[comptime] served: usize,
    #[comptime] spread: usize,
    #[comptime] accumulator_width: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    comptime!(assert_spread(
        served,
        spread,
        accumulator_width,
        "block::seed"
    ));
    let guard = comptime!(spread_guard(spread, cols));
    let monoid = acc.monoid();
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            if comptime!(spread > 1) {
                let base = (n as u32).fmul(comptime!(spread as u32));
                // The spare lanes of an overhanging last column have no cell to seed from, and
                // the identity they keep contributes nothing to the fold.
                let mut lanes = Vector::<E, V>::cast_from(Monoid::identity::<E>(monoid));
                #[unroll]
                for l in 0..spread {
                    let col = base.fadd(comptime!(l as u32));
                    let live = if comptime!(guard) {
                        col < comptime!(cols as u32)
                    } else {
                        true.runtime()
                    };
                    if live {
                        lanes.insert(l, acc.seed((i as u32, col)).extract(0usize));
                    }
                }
                c[i * nr + n] = lanes;
            } else {
                let cell = acc.seed((i as u32, n as u32));
                if comptime!(served > 1) {
                    let mut lanes = Vector::<E, V>::cast_from(Monoid::identity::<E>(monoid));
                    lanes.insert(0usize, cell.extract(0usize));
                    c[i * nr + n] = lanes;
                } else {
                    c[i * nr + n] = Vector::<E, V>::cast_from(cell);
                }
            }
        }
    }
    c
}

/// The twin of [`seed`]: commit the block back once the whole contraction is folded into it,
/// collapsing the block's lanes first where they hold partials of one cell (`served > 1`), or
/// scattering them across scalar sink cells where they hold neighbours (`spread > 1`).
/// Through [`AccumulateView`], so a lane-split accumulator reduces across lanes on the way out
/// rather than the leaf knowing it was split.
#[cube]
pub(crate) fn commit<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    c: Array<Vector<E, V>>,
    #[comptime] served: usize,
    #[comptime] spread: usize,
    #[comptime] accumulator_width: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] unroll: bool,
) {
    comptime!(assert_spread(
        served,
        spread,
        accumulator_width,
        "block::commit"
    ));
    let guard = comptime!(spread_guard(spread, cols));
    let lane_share = acc.lane_share();
    let monoid = acc.monoid();
    comptime!(assert!(
        !guard || matches!(lane_share, LaneShare::Whole),
        "block::commit: a spread block skips the lanes overhanging the sink, and a lane-split \
         accumulator ({lane_share:?}) folds across the plane on the way out, which that skip \
         would put under divergent control flow"
    ));
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            let cell = c[i * nr + n];
            if comptime!(spread > 1) {
                let base = (n as u32).fmul(comptime!(spread as u32));
                // One commit per lane, which the assert above holds to `LaneShare::Whole`: each
                // is a bare write, not `spread` plane folds where the plain path does one.
                #[unroll]
                for l in 0..spread {
                    let col = base.fadd(comptime!(l as u32));
                    let live = if comptime!(guard) {
                        col < comptime!(cols as u32)
                    } else {
                        true.runtime()
                    };
                    if live {
                        acc.commit((i as u32, col), Vector::<E, A>::cast_from(cell.extract(l)));
                    }
                }
            } else if comptime!(served > 1) {
                let total = horizontal::vector::<E, V>(cell, served, monoid);
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(total));
            } else {
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(cell));
            }
        }
    }
}
