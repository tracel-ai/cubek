//! Cell-major gather for expensive procedural filters separable over the contracted axes.

use cubecl::prelude::*;

use crate::microkernel::block;
use crate::*;

use super::coords::{cell_position, cell_read};

/// Cache each factor's 1-D tap walk before consuming their Cartesian product for one cell.
///
/// The walk is cached per accumulator row unless `lhs_spans_col`: with no factor reading the
/// accumulator's innermost axis the weights cannot vary along it, so evaluating them per cell
/// repeats one identical walk `nr` times.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric, IR: Numeric, WPR: Size, V: Size>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] reduce_extents: Vec<usize>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] vw: usize,
    #[comptime] lhs_spans_col: bool,
    #[comptime] config: MemoryMmaConfig,
) {
    let rank = comptime!(space.rank());
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());
    let rhs_view = rhs.nd::<IR, WPR, V>();
    let rhs_check = comptime!(rhs_view.check);
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
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= config.unroll_limit && !rhs_check && !acc_check);
        // A comptime `p` folds the tap coordinates, the operand coordinate resolution and the
        // weight indices, which is what lets the walk stay in registers and vectorize. It costs
        // `kc` bodies per cell, so the limit reads the whole emitted block rather than the cell
        // count the register block is sized against.
        let unroll_taps =
            comptime!(mr * nr * kc <= config.unroll_limit && !rhs_check && !acc_check);
        let mut c = block::seed(&mut acc, comptime!(mr), comptime!(nr), unroll);

        #[unroll(unroll)]
        for i in 0..mr {
            let mut weights = Array::<EL>::new(taps);
            if comptime!(!lhs_spans_col) {
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
            }

            #[unroll(unroll)]
            for n in 0..nr {
                if comptime!(lhs_spans_col) {
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
                }

                #[unroll(unroll_taps)]
                for p in 0..kc {
                    let reduce_coords =
                        tap_coords(p.fcast::<u32>(), comptime!(reduce_extents.clone()));
                    let mut weight =
                        weights[factor_tap(&reduce_coords, 0usize, comptime!(offsets[0]))];
                    #[unroll]
                    for f in 1..factors {
                        weight *= weights[factor_tap(&reduce_coords, f, comptime!(offsets[f]))];
                    }
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
                    c[i * nr + n] = fma(Vector::<E, V>::cast_from(weight), value, c[i * nr + n]);
                }
            }
        }
        block::commit(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
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
