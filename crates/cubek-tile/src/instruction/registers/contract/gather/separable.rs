//! Cell-major gather for expensive procedural filters separable over the contracted axes.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::instruction::registers::block;
use crate::*;

use super::{
    FactorReuse, GatherProblem,
    coords::{cell_position, offset_last},
};

/// Cache each factor's 1-D tap walk at its maximal reuse level before consuming the Cartesian
/// product: once per block, row, column or cell. Recipe coordinate dependencies decide where the
/// factor itself varies; masked normalization adds any accumulator axes that also move the
/// physical bound checked for that factor.
///
/// Whether any factor varies over the innermost accumulator axis decides the contraction nesting.
/// Where none does, a tap's factor product is invariant along the output line, so taps stay outside
/// lines and the rhs position is resolved once per tap. Otherwise lines stay outside taps, while
/// row- and column-local factor caches still avoid repeating the orthogonal factor walk.
///
/// Both nestings fold the map by hand ([`Tile::nd_split`]) rather than re-running it per read. The
/// lines of one run are adjacent on the operand's innermost physical axis, which is the
/// accumulator's own column axis at coefficient `1`
/// ([`assert_separable_shapes`](super::coords::assert_separable_shapes)), so their source
/// coordinates differ in that axis alone and one cell apart. The taps above them move only the
/// contracted axes, which a resampling map steps outside its floor. When every factor is free of
/// the column axis, the whole run shares one anchor ([`AxisProjection::anchor`]) per row; otherwise
/// each `(i, n)` cell anchors once and steps its taps via
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
    let caches_block = comptime!(problem.factor_reuse.contains(&FactorReuse::Block));
    let caches_column = comptime!(problem.factor_reuse.contains(&FactorReuse::Column));
    let factors_span_col = comptime!(
        problem
            .factor_reuse
            .iter()
            .any(|reuse| matches!(reuse, FactorReuse::Column | FactorReuse::Cell))
    );

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

        let mut block_weights = Array::<EL>::new(taps);
        if comptime!(caches_block) {
            let block_anchor = rhs_anchor(
                &rhs_reader.map,
                &batch,
                0u32,
                0u32,
                comptime!(problem.clone()),
            );
            #[unroll]
            for f in 0..factors {
                if comptime!(problem.factor_reuse[f] == FactorReuse::Block) {
                    factor_walk::<EL, ER>(
                        &mut block_weights,
                        comptime!(problem.offsets[f]),
                        lhs,
                        rhs,
                        &rhs_reader.map,
                        &block_anchor,
                        &batch,
                        0u32,
                        0u32,
                        f,
                        comptime!(problem.clone()),
                    );
                }
            }
        }

        let mut column_weights = Array::<EL>::new(taps * nr);
        if comptime!(caches_column) {
            #[unroll(unroll)]
            for n in 0..nr {
                let anchor = rhs_anchor(
                    &rhs_reader.map,
                    &batch,
                    0u32,
                    n as u32,
                    comptime!(problem.clone()),
                );
                #[unroll]
                for f in 0..factors {
                    if comptime!(problem.factor_reuse[f] == FactorReuse::Column) {
                        factor_walk::<EL, ER>(
                            &mut column_weights,
                            n * comptime!(taps) + comptime!(problem.offsets[f]),
                            lhs,
                            rhs,
                            &rhs_reader.map,
                            &anchor,
                            &batch,
                            0u32,
                            n as u32,
                            f,
                            comptime!(problem.clone()),
                        );
                    }
                }
            }
        }

        #[unroll(unroll)]
        for i in 0..mr {
            let mut row_weights = Array::<EL>::new(taps);
            let row_anchor = rhs_anchor(
                &rhs_reader.map,
                &batch,
                i as u32,
                0u32,
                comptime!(problem.clone()),
            );
            #[unroll]
            for f in 0..factors {
                if comptime!(problem.factor_reuse[f] == FactorReuse::Row) {
                    factor_walk::<EL, ER>(
                        &mut row_weights,
                        comptime!(problem.offsets[f]),
                        lhs,
                        rhs,
                        &rhs_reader.map,
                        &row_anchor,
                        &batch,
                        i as u32,
                        0u32,
                        f,
                        comptime!(problem.clone()),
                    );
                }
            }
            if comptime!(factors_span_col) {
                #[unroll(unroll)]
                for n in 0..nr {
                    let anchor = rhs_anchor(
                        &rhs_reader.map,
                        &batch,
                        i as u32,
                        n as u32,
                        comptime!(problem.clone()),
                    );
                    let mut cell_weights = Array::<EL>::new(taps);
                    #[unroll]
                    for f in 0..factors {
                        if comptime!(problem.factor_reuse[f] == FactorReuse::Cell) {
                            factor_walk::<EL, ER>(
                                &mut cell_weights,
                                comptime!(problem.offsets[f]),
                                lhs,
                                rhs,
                                &rhs_reader.map,
                                &anchor,
                                &batch,
                                i as u32,
                                n as u32,
                                f,
                                comptime!(problem.clone()),
                            );
                        }
                    }

                    #[unroll(unroll_taps)]
                    for p in 0..kc {
                        let reduce_coords = unravel_const(
                            comptime!(problem.block.reduce_extents.clone()),
                            p.fcast::<u32>(),
                        );
                        let weight = tap_weight::<EL>(
                            &block_weights,
                            &row_weights,
                            &column_weights,
                            &cell_weights,
                            &reduce_coords,
                            n,
                            comptime!(factors),
                            comptime!(taps),
                            comptime!(problem.offsets.clone()),
                            comptime!(problem.factor_reuse.clone()),
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
                #[unroll(unroll_taps)]
                for p in 0..kc {
                    let reduce_coords = unravel_const(
                        comptime!(problem.block.reduce_extents.clone()),
                        p.fcast::<u32>(),
                    );
                    let weight = Vector::<E, V>::cast_from(row_cached_tap_weight::<EL>(
                        &block_weights,
                        &row_weights,
                        &reduce_coords,
                        comptime!(factors),
                        comptime!(problem.offsets.clone()),
                        comptime!(problem.factor_reuse.clone()),
                    ));

                    let base = rhs_reader.map.advance(
                        &row_anchor,
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

/// Anchor the rhs projection at one accumulator cell with every contracted coordinate at zero.
#[cube]
fn rhs_anchor(
    rhs_map: &AxisProjection,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] problem: GatherProblem,
) -> CoordsDyn {
    rhs_map.anchor(
        cell_position(
            batch,
            row,
            col,
            &factor_coords(comptime!(problem.factors.unwrap()), 0usize, 0usize),
            comptime!(problem.rhs_space.clone()),
            comptime!(problem.clone()),
            comptime!(problem.block.vw),
        ),
        comptime!(problem.block.reduce.clone()),
    )
}

/// Evaluate one factor's complete 1-D tap walk at the accumulator coordinate where it is cached.
#[cube]
#[allow(clippy::too_many_arguments)]
fn factor_walk<EL: Numeric, ER: Numeric>(
    weights: &mut Array<EL>,
    offset: usize,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    rhs_map: &AxisProjection,
    anchor: &CoordsDyn,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] factor: usize,
    #[comptime] problem: GatherProblem,
) {
    match comptime!(problem.normalization) {
        None =>
        {
            #[unroll]
            for k in 0..comptime!(problem.block.reduce_extents[factor]) {
                let pos = cell_position(
                    batch,
                    row,
                    col,
                    &factor_coords(comptime!(problem.factors.unwrap()), factor, k),
                    comptime!(problem.lhs_space.clone()),
                    comptime!(problem.clone()),
                    1usize,
                );
                weights[offset + comptime!(k)] = lhs.separable_factor(pos, factor);
            }
        }
        Some((mask, guard)) => {
            let mut sum = EL::from_int(0);
            #[unroll]
            for k in 0..comptime!(problem.block.reduce_extents[factor]) {
                let reduce_coords = factor_coords(comptime!(problem.factors.unwrap()), factor, k);
                let lhs_pos = cell_position(
                    batch,
                    row,
                    col,
                    &reduce_coords,
                    comptime!(problem.lhs_space.clone()),
                    comptime!(problem.clone()),
                    1usize,
                );
                let weight = lhs.separable_factor(lhs_pos, factor);
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
                                comptime!(problem.block.reduce[factor]),
                            ),
                            weight,
                            EL::from_int(0),
                        )
                    }
                    TapMask::Unmasked => weight,
                };
                weights[offset + comptime!(k)] = weight;
                sum += weight;
            }

            let reciprocal = guarded_recip_numeric::<EL>(sum, guard);
            #[unroll]
            for k in 0..comptime!(problem.block.reduce_extents[factor]) {
                weights[offset + comptime!(k)] *= reciprocal;
            }
        }
    }
}

/// Fold one tap's per-factor weights from their maximal cache levels.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tap_weight<EL: Numeric>(
    block: &Array<EL>,
    row: &Array<EL>,
    column: &Array<EL>,
    cell: &Array<EL>,
    reduce_coords: &Coords<u32>,
    column_index: usize,
    #[comptime] factors: usize,
    #[comptime] taps: usize,
    #[comptime] offsets: Vec<usize>,
    #[comptime] reuse: Vec<FactorReuse>,
) -> EL {
    let mut weight = factor_weight(
        block,
        row,
        column,
        cell,
        reduce_coords,
        column_index,
        0usize,
        taps,
        comptime!(offsets[0]),
        comptime!(reuse[0]),
    );
    #[unroll]
    for f in 1..factors {
        weight *= factor_weight(
            block,
            row,
            column,
            cell,
            reduce_coords,
            column_index,
            f,
            taps,
            comptime!(offsets[f]),
            comptime!(reuse[f]),
        );
    }
    weight
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn factor_weight<EL: Numeric>(
    block: &Array<EL>,
    row: &Array<EL>,
    column: &Array<EL>,
    cell: &Array<EL>,
    reduce_coords: &Coords<u32>,
    column_index: usize,
    #[comptime] factor: usize,
    #[comptime] taps: usize,
    #[comptime] offset: usize,
    #[comptime] reuse: FactorReuse,
) -> EL {
    let tap = factor_tap(reduce_coords, factor, offset);
    match comptime!(reuse) {
        FactorReuse::Block => block[tap],
        FactorReuse::Row => row[tap],
        FactorReuse::Column => column[column_index * comptime!(taps) + tap],
        FactorReuse::Cell => cell[tap],
    }
}

/// The column-free nesting only needs block- and row-cached factors.
#[cube]
fn row_cached_tap_weight<EL: Numeric>(
    block: &Array<EL>,
    row: &Array<EL>,
    reduce_coords: &Coords<u32>,
    #[comptime] factors: usize,
    #[comptime] offsets: Vec<usize>,
    #[comptime] reuse: Vec<FactorReuse>,
) -> EL {
    let first = factor_tap(reduce_coords, 0usize, comptime!(offsets[0]));
    let mut weight = match comptime!(reuse[0]) {
        FactorReuse::Block => block[first],
        FactorReuse::Row => row[first],
        FactorReuse::Column | FactorReuse::Cell => unreachable!(),
    };
    #[unroll]
    for f in 1..factors {
        let tap = factor_tap(reduce_coords, f, comptime!(offsets[f]));
        weight *= match comptime!(reuse[f]) {
            FactorReuse::Block => block[tap],
            FactorReuse::Row => row[tap],
            FactorReuse::Column | FactorReuse::Cell => unreachable!(),
        };
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
