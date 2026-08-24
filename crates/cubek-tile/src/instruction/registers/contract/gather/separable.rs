//! Cell-major gather for expensive procedural filters separable over the contracted axes.

use cubecl::prelude::*;

use crate::instruction::registers::block;
use crate::*;

use cubecl::std::tensor::layout::CoordsDyn;

use super::coords::{assert_operand_shapes, cell_position, cell_read};

/// Cache each factor's 1-D tap walk before consuming their Cartesian product for one cell.
///
/// The walk is cached per accumulator row unless `lhs_spans_col`: with no factor reading the
/// accumulator's innermost axis the weights cannot vary along it, so evaluating them per cell
/// repeats one identical walk `nr` times.
///
/// That same condition decides the nesting. Where no factor reads the innermost axis, a tap's
/// coordinate and its factor product are invariant along it, so the taps go outside the lines and
/// each is resolved once per tap rather than `nr` times. The schedule's cost is per tap, so that
/// factor is the whole gap against a walk that hoists them by hand. A spanning lhs needs its walk
/// per line, which has to stay outside the taps, so it nests the other way.
///
/// Both nestings fold the map by hand ([`Tile::nd_map`]) rather than re-running it per read. The
/// lines of one run are adjacent on the operand's innermost physical axis, which is one logical
/// axis at coefficient `1` ([`Projection::validate`](crate::Projection::validate)), so their source
/// coordinates differ in that axis alone and one cell apart. The taps above them move only the
/// contracted axes, which a resampling map steps outside its floor, so the whole run shares one
/// anchor ([`AxisProjection::anchor`]) and each read is that anchor plus an addition. Re-running
/// the map would spell every term again, and under a rational axis a divide with them, per line
/// and per tap.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric, V: Size>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] vw: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] replace: bool,
) {
    let rank = comptime!(space.rank());
    let (mr, nr) = comptime!((space.extent_at(rank - 2), space.extent_at(rank - 1) / vw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());

    // Contracted extents belong to the merged operands: those axes are absent from the sink.
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let reduce = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space).to_vec());
    let reduce_extents = comptime!(reduce.iter().map(|&a| merged.extent(a)).collect::<Vec<_>>());
    let lhs_spans_col = comptime!(lhs.space.contains(space.axis_at(rank - 1)));

    comptime!(assert_operand_shapes(
        &lhs.space,
        &rhs.space,
        &space,
        &reduce,
        vw,
        lhs_spans_col
    ));
    let factor_count = lhs.factors();
    comptime!(assert!(
        factor_count == reduce.len(),
        "contract gather: a separable lhs needs one factor per contracted axis"
    ));

    let rhs_view = rhs.nd_packed::<V>();
    let rhs_map = rhs.nd_map();
    let rhs_box = rhs.nd_physical_packed::<V>();
    let phys_rank = rhs.physical_rank();
    let factors = comptime!(reduce_extents.len());
    let kc = comptime!(reduce_extents.iter().product::<usize>());
    let offsets = comptime!(tap_offsets(&reduce_extents));
    let taps = comptime!(reduce_extents.iter().sum::<usize>());
    let batch_extents = const_coords(comptime!(
        (0..rank - 2)
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    ));

    for mat in 0..matrices {
        let batch = unravel(&batch_extents, mat.fcast::<u32>());
        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));
        // A comptime `p` folds the tap coordinates, the operand coordinate resolution and the
        // weight indices, which is what lets the walk stay in registers and vectorize. It costs
        // `kc` bodies per cell, so the taps unroll on their own budget: the whole nest only when
        // every cell's scalars fit in it too.
        let unroll = comptime!(mr * nr * vw * kc <= config.budget);
        let unroll_taps = comptime!(kc <= config.budget);
        let mut c = block::seed::<E, V, V>(
            &mut acc,
            1usize,
            comptime!(mr),
            comptime!(nr),
            unroll,
            replace,
        );

        #[unroll(unroll)]
        for i in 0..mr {
            if comptime!(lhs_spans_col) {
                #[unroll(unroll)]
                for n in 0..nr {
                    let mut weights = Array::<EL>::new(taps);
                    tap_walk::<EL>(
                        &mut weights,
                        lhs,
                        &batch,
                        i as u32,
                        n as u32,
                        comptime!(space.clone()),
                        comptime!(reduce.clone()),
                        comptime!(reduce_extents.clone()),
                        comptime!(offsets.clone()),
                    );

                    #[unroll(unroll_taps)]
                    for p in 0..kc {
                        let reduce_coords =
                            tap_coords(p.fcast::<u32>(), comptime!(reduce_extents.clone()));
                        let weight = tap_weight::<EL>(
                            &weights,
                            &reduce_coords,
                            comptime!(factors),
                            comptime!(offsets.clone()),
                        );
                        let value = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                            &rhs_view,
                            &batch,
                            i as u32,
                            n as u32,
                            &reduce_coords,
                            comptime!(rhs.space.clone()),
                            comptime!(space.clone()),
                            comptime!(reduce.clone()),
                            vw,
                        ));
                        c[i * nr + n] =
                            fma(Vector::<E, V>::cast_from(weight), value, c[i * nr + n]);
                    }
                }
            } else {
                let mut weights = Array::<EL>::new(taps);
                tap_walk::<EL>(
                    &mut weights,
                    lhs,
                    &batch,
                    i as u32,
                    0u32,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(offsets.clone()),
                );

                // The taps are the only coordinates moving under this row, so the map's rational
                // axes carry the same numerator at all `kc` of them: their floor is taken once
                // here and each tap steps the result.
                let anchor = rhs_map.anchor(
                    cell_position(
                        &batch,
                        i as u32,
                        0u32,
                        &factor_coords(comptime!(factors), 0usize, 0usize),
                        comptime!(rhs.space.clone()),
                        comptime!(space.clone()),
                        comptime!(reduce.clone()),
                        vw,
                    ),
                    comptime!(reduce.clone()),
                );

                #[unroll(unroll_taps)]
                for p in 0..kc {
                    let reduce_coords =
                        tap_coords(p.fcast::<u32>(), comptime!(reduce_extents.clone()));
                    let weight = Vector::<E, V>::cast_from(tap_weight::<EL>(
                        &weights,
                        &reduce_coords,
                        comptime!(factors),
                        comptime!(offsets.clone()),
                    ));

                    let base = rhs_map.advance(
                        &anchor,
                        cell_position(
                            &batch,
                            i as u32,
                            0u32,
                            &reduce_coords,
                            comptime!(rhs.space.clone()),
                            comptime!(space.clone()),
                            comptime!(reduce.clone()),
                            vw,
                        ),
                        comptime!(reduce.clone()),
                    );

                    #[unroll(unroll)]
                    for n in 0..nr {
                        let value = Vector::<E, V>::cast_from(rhs_box.read(step_line(
                            &base,
                            comptime!(phys_rank),
                            n as u32,
                        )));
                        c[i * nr + n] = fma(weight, value, c[i * nr + n]);
                    }
                }
            }
        }
        block::commit::<E, V, V>(&mut acc, c, 1usize, comptime!(mr), comptime!(nr), unroll);
    }
}

/// Evaluate every factor's 1-D tap walk at one accumulator cell.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tap_walk<EL: Numeric>(
    weights: &mut Array<EL>,
    lhs: &Tile<EL>,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] space: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] reduce_extents: Vec<usize>,
    #[comptime] offsets: Vec<usize>,
) {
    let factors = comptime!(reduce_extents.len());

    #[unroll]
    for f in 0..factors {
        #[unroll]
        for k in 0..comptime!(reduce_extents[f]) {
            let pos = cell_position(
                batch,
                row,
                col,
                &factor_coords(comptime!(factors), f, k),
                comptime!(lhs.space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                1usize,
            );
            weights[comptime!(offsets[f] + k)] = lhs.separable_factor(pos, f);
        }
    }
}

/// `base` advanced by `step` cells along the innermost physical axis, the only one a run of
/// adjacent lines moves.
#[cube]
fn step_line(base: &CoordsDyn, #[comptime] rank: usize, step: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();

    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(base[p].fadd(step));
        } else {
            out.push(base[p]);
        }
    }

    out
}

fn tap_offsets(extents: &[usize]) -> Vec<usize> {
    extents
        .iter()
        .scan(0, |start, taps| {
            let at = *start;
            *start += taps;
            Some(at)
        })
        .collect()
}

/// Fold one tap's per-factor weights into the product its cell accumulates.
#[cube]
fn tap_weight<EL: Numeric>(
    weights: &Array<EL>,
    reduce_coords: &Coords<u32>,
    #[comptime] factors: usize,
    #[comptime] offsets: Vec<usize>,
) -> EL {
    let mut weight = weights[factor_tap(reduce_coords, 0usize, comptime!(offsets[0]))];
    #[unroll]
    for f in 1..factors {
        weight *= weights[factor_tap(reduce_coords, f, comptime!(offsets[f]))];
    }
    weight
}

#[cube]
fn factor_tap(
    reduce_coords: &Coords<u32>,
    #[comptime] factor: usize,
    #[comptime] offset: usize,
) -> usize {
    reduce_coords
        .at(factor)
        .fadd(comptime!(offset as u32))
        .fcast::<usize>()
}

#[cube]
fn factor_coords(
    #[comptime] factors: usize,
    #[comptime] factor: usize,
    #[comptime] tap: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for f in 0..factors {
        out.push(comptime!(if f == factor { tap as u32 } else { 0u32 }));
    }

    out
}

/// Resolve a flat Cartesian-product tap into one coordinate per factor.
#[cube]
fn tap_coords(p: u32, #[comptime] extents: Vec<usize>) -> Coords<u32> {
    let n = comptime!(extents.len());
    let mut out = Coords::<u32>::new();

    #[unroll]
    for f in 0..n {
        let digit = p.fdiv(comptime!(extents[f + 1..].iter().product::<usize>() as u32));
        if comptime!(f == 0) {
            out.push(digit);
        } else {
            out.push(digit.frem(comptime!(extents[f] as u32)));
        }
    }

    out
}
