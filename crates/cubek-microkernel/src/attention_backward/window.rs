use crate::matmul::MatmulAxes;
use crate::matmul::dense::plane::PlaneFragmentMicrokernel;
use cubecl::tile::{Axis, Level, Region, Semiring, Space, StageStorage, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// A plane's windows of shared memory between a step's contractions: its scores and the gradient
/// against its probabilities, summed in `EA`, and the one of them the next contraction loads, cast
/// to the staged element. The plane meets on `sync_plane` around each.
#[derive(CubeType, Clone)]
pub(crate) struct ScoreWindows {
    pub scores: Tile,
    pub gradients: Tile,
    pub cast: Tile,
}

#[cube]
impl ScoreWindows {
    /// The windows of one step of `walk`, laid out along `axes`: every plane's the step holds.
    pub fn open<EA: Float, ES: Float>(walk: &Walk, #[comptime] axes: Vec<Axis>) -> ScoreWindows {
        ScoreWindows {
            scores: Tile::scratch::<EA>(walk, comptime!(axes.clone()), StageStorage::Strided),
            gradients: Tile::scratch::<EA>(walk, comptime!(axes.clone()), StageStorage::Strided),
            cast: Tile::scratch::<ES>(walk, axes, StageStorage::Strided),
        }
    }

    /// `plane`'s windows.
    pub fn at(&self, plane: &Region) -> ScoreWindows {
        ScoreWindows {
            scores: self.scores.at(plane),
            gradients: self.gradients.at(plane),
            cast: self.cast.at(plane),
        }
    }
}

/// `window = lhs · rhs`, summed in `EA` on `product`'s own grid of fragments, placed as `levels`
/// state it below the walker's planes, then cast into the plane's `window` of shared memory: where
/// a plane keeps a score, or the gradient against its probabilities, between two contractions.
/// Each fragment of the grid pairs the operands' fragments by position: the lhs's rows by steps of
/// the contraction, the rhs's columns by the same steps. The caller meets the plane before the
/// window is read.
#[cube]
pub(crate) fn contract_into_window<EA: Float>(
    window: &Tile,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] product: PlaneFragmentMicrokernel,
    #[comptime] levels: Vec<Level>,
) {
    // The accumulator, cut to the product's own grid below the planes' level.
    let placed = window.with_levels(1usize, levels);
    let mut sum = placed.accumulator::<EA>(
        lhs,
        rhs,
        comptime!(product.tile_instruction()),
        Semiring::SUM_PROD,
    );
    sum.zero();
    let [(_, m), (_, n), (_, k)] = comptime!(product.instruction.shape());
    let MatmulAxes {
        m: rows_axis,
        n: columns_axis,
        k: depth_axis,
    } = comptime!(product.axes);
    let (rows, columns, depth) = comptime!((
        product.settings.along_m.get(),
        product.settings.along_n.get(),
        product.settings.depth.get(),
    ));
    let lhs_fragments = lhs.over(&comptime!(Level::every(&[(rows_axis, m), (depth_axis, k)])));
    let rhs_fragments = rhs.over(&comptime!(Level::every(&[
        (columns_axis, n),
        (depth_axis, k)
    ])));
    let fragment = comptime!(Level::every(&[(rows_axis, m), (columns_axis, n)]));
    let sum_fragments = sum.over(&fragment);
    #[unroll]
    for step in 0..depth {
        #[unroll]
        for row in 0..rows {
            #[unroll]
            for column in 0..columns {
                let (at_sum, at_lhs, at_rhs) = comptime!((
                    fragment_at(
                        &sum.space(),
                        (rows_axis, row, rows),
                        (columns_axis, column, columns)
                    ),
                    fragment_at(
                        &lhs.space(),
                        (rows_axis, row, rows),
                        (depth_axis, step, depth)
                    ),
                    fragment_at(
                        &rhs.space(),
                        (columns_axis, column, columns),
                        (depth_axis, step, depth)
                    ),
                ));
                let mut cells = sum.at(&sum_fragments.region(at_sum));
                cells.mma(
                    &lhs.at(&lhs_fragments.region(at_lhs)),
                    &rhs.at(&rhs_fragments.region(at_rhs)),
                );
            }
        }
    }
    // The accumulator's fragments and the window are one grid at two depths: the accumulator is
    // placed on the product's own grid, the window sits under the plane's region, and a region
    // only windows a tile at its depth.
    let window_fragments = window.over(&fragment);
    #[unroll]
    for at in 0..comptime!(rows * columns) {
        let mut cells = window.at(&window_fragments.region(at));
        cells.copy_cast_from(&sum.at(&sum_fragments.region(at)));
    }
}

/// The place among a tile's fragments of the one at `first.1` of `first.2` along `first.0` and
/// `second.1` of `second.2` along `second.0`: a level's regions run along the tile's own axes in
/// the order its space lays them, the last fastest.
fn fragment_at(
    space: &Space,
    (first, first_at, first_count): (Axis, usize, usize),
    (second, second_at, second_count): (Axis, usize, usize),
) -> usize {
    match space.position(first) < space.position(second) {
        true => first_at * second_count + second_at,
        false => second_at * first_count + first_at,
    }
}
