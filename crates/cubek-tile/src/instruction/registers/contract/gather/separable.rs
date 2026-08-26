//! Cell-major gather for expensive procedural filters separable over the contracted axes.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::instruction::registers::block;
use crate::*;

use super::{
    GatherProblem, LhsRole,
    coords::{cell_position, offset_last},
};

/// Cache each factor's 1-D tap walk before consuming their Cartesian product for one cell.
///
/// The walk is cached per accumulator row unless the lhs spans the column: with no factor
/// reading the accumulator's innermost axis the weights cannot vary along it, so evaluating them
/// per cell repeats one identical walk `nr` times.
///
/// That same condition decides the nesting. Where no factor reads the innermost axis, a tap's
/// coordinate and its factor product are invariant along it, so the taps go outside the lines and
/// each is resolved once per tap rather than `nr` times. The schedule's cost is per tap, so that
/// factor is the whole gap against a walk that hoists them by hand. A spanning lhs needs its walk
/// per line, which has to stay outside the taps, so it nests the other way.
///
/// Both nestings fold the map by hand ([`Tile::nd_split`]) rather than re-running it per read. The
/// lines of one run are adjacent on the operand's innermost physical axis, which is the
/// accumulator's own column axis at coefficient `1`
/// ([`assert_separable_shapes`](super::coords::assert_separable_shapes)), so their source
/// coordinates differ in that axis alone and one cell apart. The taps above them move only the
/// contracted axes, which a resampling map steps outside its floor. Where the lhs does not span the
/// column axis, the whole run shares one anchor ([`AxisProjection::anchor`]) per row; where it
/// spans the column axis, each `(i, n)` cell anchors once and steps its taps via
/// [`AxisProjection::advance`]. In both cases, reads and mask tests use the stepped physical
/// coordinates rather than evaluating the projection terms per tap.
#[cube]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric, V: Size, A: Size>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] problem: GatherProblem,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(problem.block.mr);
    let nr = comptime!(problem.block.nr);
    let cols = comptime!(problem.block.cols);
    let spread = comptime!(problem.block.spread);
    let aw = comptime!(problem.block.aw);
    let matrices = comptime!(problem.block.matrices());
    let batch_extents = comptime!(problem.block.batch_extents());

    let rhs_reader = rhs.nd_split_packed::<V>();
    let rhs_check = comptime!(rhs_reader.view.check);
    let factors = comptime!(
        problem
            .factors
            .expect("contract gather: the separable schedule needs a factorized lhs")
    );
    let kc = comptime!(problem.block.kc);
    let taps = comptime!(problem.taps);

    for mat in 0..matrices {
        let batch = unravel_const(comptime!(batch_extents.clone()), mat.fcast::<u32>());

        // The contraction's own algebra, as [`direct`](super::direct) states it.
        let mut acc = acc.matrix_accumulate::<A>(
            mat,
            comptime!(problem.block.space.clone()),
            comptime!(semiring.add()),
        );
        // A comptime `p` folds the tap coordinates, the operand coordinate resolution and the
        // weight indices, which is what lets the walk stay in registers and vectorize. It costs
        // `kc` bodies per cell, so the taps unroll on their own budget: the whole nest only when
        // every cell's scalars fit in it too.
        let acc_check = acc.check();
        let unroll =
            comptime!(problem.block.scalars() * kc <= config.budget && !rhs_check && !acc_check);
        let unroll_taps = comptime!(kc <= config.budget);
        let mut c = block::seed::<E, V, A>(
            &mut acc,
            1usize,
            spread,
            aw,
            comptime!(mr),
            comptime!(nr),
            comptime!(cols),
            unroll,
        );

        #[unroll(unroll)]
        for i in 0..mr {
            if comptime!(problem.lhs != LhsRole::FreeOfColumn) {
                #[unroll(unroll)]
                for n in 0..nr {
                    let anchor = rhs_reader.map.anchor(
                        cell_position(
                            &batch,
                            i as u32,
                            n as u32,
                            &factor_coords(comptime!(factors), 0usize, 0usize),
                            comptime!(problem.rhs_space.clone()),
                            comptime!(problem.clone()),
                            comptime!(problem.block.vw),
                        ),
                        comptime!(problem.block.reduce.clone()),
                    );
                    let mut weights = Array::<EL>::new(taps);
                    tap_walk::<EL, ER>(
                        &mut weights,
                        lhs,
                        rhs,
                        &rhs_reader.map,
                        &anchor,
                        &batch,
                        i as u32,
                        n as u32,
                        comptime!(factors),
                        comptime!(problem.clone()),
                    );

                    #[unroll(unroll_taps)]
                    for p in 0..kc {
                        let reduce_coords = unravel_const(
                            comptime!(problem.block.reduce_extents.clone()),
                            p.fcast::<u32>(),
                        );
                        let weight = tap_weight::<EL>(
                            &weights,
                            &reduce_coords,
                            comptime!(factors),
                            comptime!(problem.offsets.clone()),
                        );
                        let base = rhs_reader.map.advance(
                            &anchor,
                            cell_position(
                                &batch,
                                i as u32,
                                n as u32,
                                &reduce_coords,
                                comptime!(problem.rhs_space.clone()),
                                comptime!(problem.clone()),
                                comptime!(problem.block.vw),
                            ),
                            comptime!(problem.block.reduce.clone()),
                        );
                        let value = Vector::<E, V>::cast_from(rhs_reader.view.read(base));
                        // One semiring step, for the reason [`block::rank1_update`] gives.
                        c[i * nr + n] = semiring.step::<Vector<E, V>>(
                            Vector::<E, V>::cast_from(weight),
                            value,
                            c[i * nr + n],
                        );
                    }
                }
            } else {
                // The taps are the only coordinates moving under this row, so the map's rational
                // axes carry the same numerator at all `kc` of them: their floor is taken once
                // here and each tap steps the result.
                let anchor = rhs_reader.map.anchor(
                    cell_position(
                        &batch,
                        i as u32,
                        0u32,
                        &factor_coords(comptime!(factors), 0usize, 0usize),
                        comptime!(problem.rhs_space.clone()),
                        comptime!(problem.clone()),
                        comptime!(problem.block.vw),
                    ),
                    comptime!(problem.block.reduce.clone()),
                );

                let mut weights = Array::<EL>::new(taps);
                tap_walk::<EL, ER>(
                    &mut weights,
                    lhs,
                    rhs,
                    &rhs_reader.map,
                    &anchor,
                    &batch,
                    i as u32,
                    0u32,
                    comptime!(factors),
                    comptime!(problem.clone()),
                );

                #[unroll(unroll_taps)]
                for p in 0..kc {
                    let reduce_coords = unravel_const(
                        comptime!(problem.block.reduce_extents.clone()),
                        p.fcast::<u32>(),
                    );
                    let weight = Vector::<E, V>::cast_from(tap_weight::<EL>(
                        &weights,
                        &reduce_coords,
                        comptime!(factors),
                        comptime!(problem.offsets.clone()),
                    ));

                    let base = rhs_reader.map.advance(
                        &anchor,
                        cell_position(
                            &batch,
                            i as u32,
                            0u32,
                            &reduce_coords,
                            comptime!(problem.rhs_space.clone()),
                            comptime!(problem.clone()),
                            comptime!(problem.block.vw),
                        ),
                        comptime!(problem.block.reduce.clone()),
                    );

                    #[unroll(unroll)]
                    for n in 0..nr {
                        let value = Vector::<E, V>::cast_from(rhs_reader.view.read(offset_last(
                            &base,
                            comptime!(rhs_reader.rank),
                            n as u32,
                        )));
                        // One semiring step, for the reason [`block::rank1_update`] gives.
                        c[i * nr + n] = semiring.step::<Vector<E, V>>(weight, value, c[i * nr + n]);
                    }
                }
            }
        }
        block::commit::<E, V, A>(
            &mut acc,
            c,
            1usize,
            spread,
            aw,
            comptime!(mr),
            comptime!(nr),
            comptime!(cols),
            unroll,
        );
    }
}

/// Evaluate every factor's 1-D tap walk at one accumulator cell.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tap_walk<EL: Numeric, ER: Numeric>(
    weights: &mut Array<EL>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    rhs_map: &AxisProjection,
    anchor: &CoordsDyn,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] factors: usize,
    #[comptime] problem: GatherProblem,
) {
    match comptime!(problem.normalization) {
        None =>
        {
            #[unroll]
            for f in 0..factors {
                #[unroll]
                for k in 0..comptime!(problem.block.reduce_extents[f]) {
                    let pos = cell_position(
                        batch,
                        row,
                        col,
                        &factor_coords(comptime!(factors), f, k),
                        comptime!(problem.lhs_space.clone()),
                        comptime!(problem.clone()),
                        1usize,
                    );
                    weights[comptime!(problem.offsets[f] + k)] = lhs.separable_factor(pos, f);
                }
            }
        }
        Some((mask, guard)) =>
        {
            #[unroll]
            for f in 0..factors {
                let mut sum = EL::from_int(0);
                #[unroll]
                for k in 0..comptime!(problem.block.reduce_extents[f]) {
                    let reduce_coords = factor_coords(comptime!(factors), f, k);
                    let lhs_pos = cell_position(
                        batch,
                        row,
                        col,
                        &reduce_coords,
                        comptime!(problem.lhs_space.clone()),
                        comptime!(problem.clone()),
                        1usize,
                    );
                    let weight = lhs.separable_factor(lhs_pos, f);
                    let weight = match comptime!(mask) {
                        TapMask::Masked => {
                            let rhs_pos = cell_position(
                                batch,
                                row,
                                col,
                                &reduce_coords,
                                comptime!(problem.rhs_space.clone()),
                                comptime!(problem.clone()),
                                comptime!(problem.block.vw),
                            );
                            let physical_pos = rhs_map.advance(
                                anchor,
                                rhs_pos,
                                comptime!(problem.block.reduce.clone()),
                            );
                            select(
                                rhs.separable_physical_tap_in_bounds(
                                    &physical_pos,
                                    comptime!(problem.block.reduce[f]),
                                ),
                                weight,
                                EL::from_int(0),
                            )
                        }
                        TapMask::Unmasked => weight,
                    };
                    weights[comptime!(problem.offsets[f] + k)] = weight;
                    sum += weight;
                }

                let reciprocal = guarded_recip_numeric::<EL>(sum, guard);
                #[unroll]
                for k in 0..comptime!(problem.block.reduce_extents[f]) {
                    weights[comptime!(problem.offsets[f] + k)] *= reciprocal;
                }
            }
        }
    }
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
