//! The `mr × nr` register block: seed it from the accumulator, contract into it, commit it back.
//!
//! The block is a parameter, not owned here, which is the only difference between its two
//! callers: the memory-backed nest ([`contract`](super::contract)) seeds a local one and commits
//! it back per visit, while a promoted [`RegisterData`] *is* the accumulator across the walk.

use cubecl::prelude::*;

use crate::*;

/// `c += lhs · rhs` over the block, one line of the contraction at a time.
///
/// Each factor is its values' matrix and the scales riding it, looked up at every line's own
/// coordinates ([`FactorReader`]): a factor carrying none goes through as it lies, and the walk
/// is the same either way — the lines of the contraction, in order.
///
/// A step consumes [`Space::contracted_per_step`] values. Past one, both operands line along the
/// contracted axis and the block's lanes are one cell's partials, folded by [`commit`]. At one, the
/// rhs lines along the accumulator and the lhs is read lane by lane (comptime under `lane_fanout`).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    lhs_scales: &FactorReader,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    rhs_scales: &FactorReader,
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
    // Values one line holds: a folded step takes the whole line at once, an unfolded one a lane of
    // it per step.
    let width = comptime!(match folded {
        true => contracted_per_step,
        false => lw,
    });
    let lines = comptime!(kc / width);
    let tail = comptime!(kc % width);
    // Lanes as constants, which is what lets the backend fold an `mr`-row fan-out's repeated line
    // reads into one. Where the caller did not ask for that, the lane is the walk's own index.
    let fixed = comptime!(folded || (lane_fanout && lw > 1) || lw == 1);

    for line in 0..lines {
        if comptime!(folded) {
            rank1_update::<E, EL, L, ER, V>(
                lhs,
                lhs_scales,
                rhs,
                rhs_scales,
                c,
                &mut b,
                0usize,
                line as u32,
                0usize,
                comptime!(Some(0usize)),
                contracted_per_step,
                mr,
                nr,
                unroll,
                semiring,
            );
        } else if comptime!(fixed) {
            #[unroll]
            for lane in 0..lw {
                rank1_update::<E, EL, L, ER, V>(
                    lhs,
                    lhs_scales,
                    rhs,
                    rhs_scales,
                    c,
                    &mut b,
                    line * lw + lane,
                    line as u32,
                    0usize,
                    comptime!(Some(lane)),
                    contracted_per_step,
                    mr,
                    nr,
                    unroll,
                    semiring,
                );
            }
        } else {
            for lane in 0..lw {
                rank1_update::<E, EL, L, ER, V>(
                    lhs,
                    lhs_scales,
                    rhs,
                    rhs_scales,
                    c,
                    &mut b,
                    line * lw + lane,
                    line as u32,
                    lane,
                    comptime!(None),
                    contracted_per_step,
                    mr,
                    nr,
                    unroll,
                    semiring,
                );
            }
        }
    }

    // A line width that does not divide `kc` leaves a partial last line. Its lane count is
    // comptime too, so the tail is straight-line code rather than a second, dynamic walk.
    #[unroll]
    for lane in 0..tail {
        rank1_update::<E, EL, L, ER, V>(
            lhs,
            lhs_scales,
            rhs,
            rhs_scales,
            c,
            &mut b,
            comptime!(lines * width + lane),
            comptime!(lines) as u32,
            0usize,
            comptime!(Some(lane)),
            contracted_per_step,
            mr,
            nr,
            unroll,
            semiring,
        );
    }
}

/// One step `c += outer(A[:, k], B[k, :])`, at scalar contraction step `k` off the `k_line`-th
/// K-line of each lhs row, each line under the scale covering it.
///
/// At `contracted_per_step > 1` both reads are whole lines off the contracted axis, which is why
/// the rhs is addressed `(n, k_line)` there and `(k, n)` otherwise.
///
/// `fixed` names the component to take when the walk unrolled its lanes, so `extract` is a
/// constant and the backend folds the fan-out's `mr` repeated line reads into one; `None` takes
/// `lane` at runtime. `k_line` stays a parameter so each lane body sees a loop-invariant index.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    lhs_scales: &FactorReader,
    rhs: &MatrixView<'_, Vector<ER, V>>,
    rhs_scales: &FactorReader,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    k: usize,
    k_line: u32,
    lane: usize,
    #[comptime] fixed: Option<usize>,
    #[comptime] contracted_per_step: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
    #[comptime] semiring: Semiring,
) {
    if comptime!(contracted_per_step > 1) {
        // The rhs lines along the contraction with the lhs.
        #[unroll(unroll)]
        for n in 0..nr {
            let pos = (n as u32, k_line);
            // The value promotes to the accumulator's element before its scale folds in: a
            // scale is a factor of the term, and a term is formed in `E`. Folding it in the
            // operand's own element would round the scale to that element (an `i8` store
            // would see `0.5` as `0`) and wrap the product it forms.
            b[n] = rhs_scales.apply::<E, V>(Vector::<E, V>::cast_from(rhs.read(pos)), pos);
        }
    } else {
        // The rhs lines along the accumulator.
        #[unroll(unroll)]
        for n in 0..nr {
            let pos = (k as u32, n as u32);
            b[n] = rhs_scales.apply::<E, V>(Vector::<E, V>::cast_from(rhs.read(pos)), pos);
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let pos = (i as u32, k_line);
        let line = lhs_scales.apply::<E, L>(Vector::<E, L>::cast_from(lhs.read(pos)), pos);
        let a = if comptime!(contracted_per_step > 1) {
            Vector::<E, V>::cast_from(line)
        } else if comptime!(fixed.is_some()) {
            Vector::<E, V>::cast_from(line.extract(comptime!(fixed.unwrap())))
        } else {
            Vector::<E, V>::cast_from(line.extract_dynamic(lane))
        };
        #[unroll(unroll)]
        for n in 0..nr {
            // One step of the semiring, a single `fma` for the ordinary one: `+= a * b` would
            // lower to a separate mul + dependent add (no fast-math contraction on the CPU
            // backend), doubling the FP instruction count and serializing the accumulate.
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

/// Whether a spread block's lanes must be tested against the sink's extent before they touch it.
/// The N-D nest rounds `nr` up, so the last column's spare lanes address cells past `cols` exactly
/// when `spread` does not divide it, and an unchecked [`AccumulateView`] writes straight through.
///
/// The nest reads this too: those same spare lanes address a column past the *operands'* last
/// line, so a walk that has dropped its guard would read one line outside them.
pub(crate) fn spread_guard(spread: usize, cols: usize) -> bool {
    spread > 1 && !cols.is_multiple_of(spread)
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the steps
/// never touch memory. The algebra is the view's, stated where it was built.
///
/// Where a step consumes more than one, the block's lanes are partials of one cell, so its value
/// seeds lane 0 alone and the rest start at the identity.
///
/// At `spread > 1` they instead hold neighbouring cells of a scalar sink: a padded shared-memory
/// operand serves whole lines even when source and sink are scalar, so each block column gathers
/// `spread` sink cells. `cols` is the sink's innermost extent; see [`spread_guard`] for overhang.
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
                let base = (n as u32).times(comptime!(spread as u32));
                // The spare lanes of an overhanging last column have no cell to seed from, and
                // the identity they keep contributes nothing to the fold.
                let mut lanes = Vector::<E, V>::cast_from(Monoid::identity::<E>(monoid));
                #[unroll]
                for l in 0..spread {
                    let col = base.plus(comptime!(l as u32));
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

/// The twin of [`seed`]: commit the block back once the contraction is folded into it, first
/// collapsing lanes holding one cell's partials (`contracted_per_step > 1`) or scattering lanes
/// holding neighbours (`spread > 1`).
///
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
        !guard || !lane_share.folds(),
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
                let base = (n as u32).times(comptime!(spread as u32));
                // One commit per lane, which the assert above holds to an unfolded share: each
                // is a bare write, not `spread` plane folds where the plain path does one.
                #[unroll]
                for l in 0..spread {
                    let col = base.plus(comptime!(l as u32));
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
                let total = Monoid::fold_lanes::<E, V>(cell, contracted_per_step, monoid);
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(total));
            } else {
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(cell));
            }
        }
    }
}
