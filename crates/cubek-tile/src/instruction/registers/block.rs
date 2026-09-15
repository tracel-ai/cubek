//! The `mr × nr` register block: seed it from the accumulator, contract into it, commit it back.
//!
//! The block is a parameter, not something this owns, which is the only difference between its
//! two callers: the memory-backed nest ([`contract`](super::contract)) seeds a local one
//! and commits it back per visit, while a promoted [`RegisterData`] *is* the
//! accumulator and keeps it across the whole walk. How the block is contracted is the same either
//! way.

use cubecl::prelude::*;

use crate::instruction::registers::horizontal;
use crate::instruction::registers::lines::{Lines, LinesExpand, Span};
use crate::*;

/// `c += lhs · rhs` over the block, walked as the runs its scales cover.
///
/// A run is one read of the scales: [`Span::fields`] of them, each covering [`Span::lines`] lines
/// of the contraction. The walk steps runs, then fields, then the lines of a field, and builds
/// every index out of them — `line = (run · fields + field) · lines + l`. The field a line works
/// under is therefore a constant because the walk stepped it, not because anything resolved it,
/// and an operand carrying no scales is a run of one line and walks as it always did.
///
/// A step consumes [`Space::contracted_per_step`] values. Past one, both operands line along the
/// contracted axis and the block's lanes are partials of one cell that [`commit`] folds. At one,
/// the rhs lines along the accumulator and the lhs's line is taken a lane at a time: with fixed
/// comptime extracts when `lane_fanout` (GPU), else a lane at a time at runtime.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract<
    E: Numeric,
    EL: Numeric,
    L: Size,
    ER: Numeric,
    V: Size,
    Lhs: Lines<E = EL, V = L>,
    Rhs: Lines<E = ER, V = V>,
>(
    lhs: &Lhs,
    rhs: &Rhs,
    c: &mut Array<Vector<E, V>>,
    #[comptime] lw: usize,
    #[comptime] contracted_per_step: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] lane_fanout: bool,
    #[comptime] semiring: Semiring,
) {
    let mut b = Array::<Vector<E, V>>::new(nr);
    let folded = comptime!(contracted_per_step > 1);
    let lhs_span = lhs.span();
    let rhs_span = rhs.span();
    // A folded step reads both factors along the contraction, so both group its lines. Unfolded,
    // the rhs lines along the accumulator instead, and groups the block's columns rather than
    // these lines — [`rank1_update`] walks those.
    let span = comptime!(match folded {
        true => lhs_span.join(rhs_span),
        false => lhs_span,
    });
    // Values one line holds: a folded step takes the whole line at once, an unfolded one a lane of
    // it per step.
    let width = comptime!(match folded {
        true => contracted_per_step,
        false => lw,
    });
    let lines = comptime!(kc / width);
    let tail = comptime!(kc % width);
    let run = comptime!(span.run());
    comptime!(assert!(
        lines.is_multiple_of(run),
        "block::contract: {lines} lines of a contraction do not divide into runs of {run}, so one \
         run would work under a scale that is not there; cut the contraction at a whole run"
    ));
    comptime!(assert!(
        tail == 0 || span == Span::PLAIN,
        "block::contract: a contraction of {kc} leaves {tail} values past its last whole line, \
         which no scale of a run of {run} covers; cut it at a whole line"
    ));
    // Lanes as constants, which is what lets the backend fold an `mr`-row fan-out's repeated line
    // reads into one. Where the caller did not ask for that, the lane is the walk's own index.
    let fixed = comptime!(folded || (lane_fanout && lw > 1) || lw == 1);

    for r in 0..comptime!(lines / run) {
        #[unroll]
        for field in 0..comptime!(span.fields) {
            for l in 0..comptime!(span.lines) {
                let line = r * comptime!(run) + comptime!(field * span.lines) + l;
                if comptime!(folded) {
                    rank1_update::<E, EL, L, ER, V, Lhs, Rhs>(
                        lhs,
                        rhs,
                        c,
                        &mut b,
                        0usize,
                        line as u32,
                        0usize,
                        r as u32,
                        comptime!(Some(0usize)),
                        field,
                        contracted_per_step,
                        mr,
                        nr,
                        unroll,
                        semiring,
                    );
                } else if comptime!(fixed) {
                    #[unroll]
                    for lane in 0..lw {
                        rank1_update::<E, EL, L, ER, V, Lhs, Rhs>(
                            lhs,
                            rhs,
                            c,
                            &mut b,
                            line * lw + lane,
                            line as u32,
                            0usize,
                            r as u32,
                            comptime!(Some(lane)),
                            field,
                            contracted_per_step,
                            mr,
                            nr,
                            unroll,
                            semiring,
                        );
                    }
                } else {
                    for lane in 0..lw {
                        rank1_update::<E, EL, L, ER, V, Lhs, Rhs>(
                            lhs,
                            rhs,
                            c,
                            &mut b,
                            line * lw + lane,
                            line as u32,
                            lane,
                            r as u32,
                            comptime!(None),
                            field,
                            contracted_per_step,
                            mr,
                            nr,
                            unroll,
                            semiring,
                        );
                    }
                }
            }
        }
    }

    // A line width that does not divide `kc` leaves a partial last line, which the assert above
    // holds to a factor with no scales. Its lane count is comptime too, so the tail is
    // straight-line code rather than a second, dynamic walk.
    #[unroll]
    for lane in 0..tail {
        rank1_update::<E, EL, L, ER, V, Lhs, Rhs>(
            lhs,
            rhs,
            c,
            &mut b,
            comptime!(lines * width + lane),
            comptime!(lines) as u32,
            0usize,
            0u32,
            comptime!(Some(lane)),
            0usize,
            contracted_per_step,
            mr,
            nr,
            unroll,
            semiring,
        );
    }
}

/// One step `c += outer(A[:, k], B[k, :])`, at scalar contraction step `k` off the `k_line`-th
/// K-line of each lhs row, under the scales of run `run`.
///
/// At `contracted_per_step > 1` both reads are whole lines off the contracted axis, which is why
/// the rhs is addressed `(n, k_line)` there and `(k, n)` otherwise — and why its scales are the
/// run's there and the columns' here.
///
/// `fixed` names the component to take when the walk unrolled its lanes, so `extract` names a
/// constant and the backend folds the fan-out's `mr` repeated line reads into one; `None` takes
/// `lane` at runtime. `k_line` stays a parameter so each lane body sees one loop-invariant line
/// index.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<
    E: Numeric,
    EL: Numeric,
    L: Size,
    ER: Numeric,
    V: Size,
    Lhs: Lines<E = EL, V = L>,
    Rhs: Lines<E = ER, V = V>,
>(
    lhs: &Lhs,
    rhs: &Rhs,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    k: usize,
    k_line: u32,
    lane: usize,
    run: u32,
    #[comptime] fixed: Option<usize>,
    #[comptime] field: usize,
    #[comptime] contracted_per_step: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
    #[comptime] semiring: Semiring,
) {
    if comptime!(contracted_per_step > 1) {
        // The rhs lines along the contraction with the lhs, so it works under the same run and the
        // same field of it.
        #[unroll(unroll)]
        for n in 0..nr {
            let line = rhs.line((n as u32, k_line));
            b[n] = Vector::<E, V>::cast_from(rhs.fold((n as u32, run)).apply::<ER, V>(line, field));
        }
    } else {
        // The rhs lines along the accumulator, so its scales group the block's columns: a run of
        // them is one read of its scales, and which field a column takes is this walk's to build.
        let span = rhs.span();
        if comptime!(span == Span::PLAIN) {
            // Ungrouped columns walk as the block's budget decided: a scale, where there is one,
            // covers exactly this column of this line.
            #[unroll(unroll)]
            for n in 0..nr {
                let line = rhs.line((k as u32, n as u32));
                b[n] = Vector::<E, V>::cast_from(
                    rhs.fold((k as u32, n as u32)).apply::<ER, V>(line, 0usize),
                );
            }
        } else {
            comptime!(assert!(
                nr.is_multiple_of(span.run()),
                "block::contract: {nr} columns of a block do not divide into runs of {} scales, so \
                 one run would work under a scale that is not there",
                span.run()
            ));
            #[unroll]
            for column_run in 0..comptime!(nr / span.run()) {
                #[unroll]
                for column_field in 0..comptime!(span.fields) {
                    #[unroll]
                    for l in 0..comptime!(span.lines) {
                        let n = comptime!(column_run * span.run() + column_field * span.lines + l);
                        let line = rhs.line((k as u32, comptime!(n as u32).runtime()));
                        b[n] = Vector::<E, V>::cast_from(
                            rhs.fold((k as u32, comptime!(column_run as u32).runtime()))
                                .apply::<ER, V>(line, column_field),
                        );
                    }
                }
            }
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let line = lhs
            .fold((i as u32, run))
            .apply::<EL, L>(lhs.line((i as u32, k_line)), field);
        let a = if comptime!(contracted_per_step > 1) {
            Vector::<E, V>::cast_from(line)
        } else if comptime!(fixed.is_some()) {
            Vector::<E, V>::cast_from(line.extract(comptime!(fixed.unwrap())))
        } else {
            Vector::<E, V>::cast_from(line.extract_dynamic(lane))
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
fn assert_spread(contracted_per_step: usize, spread: usize, accumulator_width: usize, who: &str) {
    assert!(
        contracted_per_step == 1 || spread == 1,
        "{who}: lanes cannot hold contracted partials (contracted_per_step {contracted_per_step}) and neighbouring \
         sink cells (spread {spread}) at once"
    );
    assert!(
        spread == 1 || accumulator_width == 1,
        "{who}: a spread block scatters one lane per sink cell, so the accumulator must be \
         contracted_per_step scalar (it is {accumulator_width} wide)"
    );
}

/// Whether a spread block's lanes have to be tested against the sink's extent before they touch
/// it. The N-D nest rounds `nr` up, so the last column's spare lanes address cells past `cols`
/// exactly when `spread` does not divide it. Nothing masks them downstream: an unchecked
/// [`AccumulateView`] writes straight through.
///
/// The nest reads this too: those same spare lanes address a column past the *operands'* last
/// line, so a walk that has dropped its guard would read one line outside them.
pub(crate) fn spread_guard(spread: usize, cols: usize) -> bool {
    spread > 1 && !cols.is_multiple_of(spread)
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the steps
/// never touch memory. The algebra is the view's, stated where it was built.
///
/// Where a step consumes more than one, the block's lanes are partials of one cell, so its value seeds
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
    #[comptime] contracted_per_step: usize,
    #[comptime] spread: usize,
    #[comptime] accumulator_width: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    comptime!(assert_spread(
        contracted_per_step,
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
                if comptime!(contracted_per_step > 1) {
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
/// collapsing the block's lanes first where they hold partials of one cell (`contracted_per_step > 1`), or
/// scattering them across scalar sink cells where they hold neighbours (`spread > 1`).
/// Through [`AccumulateView`], so a lane-split accumulator reduces across lanes on the way out
/// rather than the leaf knowing it was split.
#[cube]
pub(crate) fn commit<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    c: Array<Vector<E, V>>,
    #[comptime] contracted_per_step: usize,
    #[comptime] spread: usize,
    #[comptime] accumulator_width: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] unroll: bool,
) {
    comptime!(assert_spread(
        contracted_per_step,
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
            } else if comptime!(contracted_per_step > 1) {
                let total = horizontal::vector::<E, V>(cell, contracted_per_step, monoid);
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(total));
            } else {
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(cell));
            }
        }
    }
}
