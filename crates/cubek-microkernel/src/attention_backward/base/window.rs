use crate::attention::plane::ScoreHolder;
use crate::attention::{Q, S};
use crate::matmul::MatmulAxes;
use crate::matmul::plane::PlaneFragmentMicrokernel;
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
            scores: Tile::scratch::<EA>(walk, axes.clone(), StageStorage::Strided),
            gradients: Tile::scratch::<EA>(walk, axes.clone(), StageStorage::Strided),
            cast: Tile::scratch::<ES>(walk, axes, StageStorage::Strided),
        }
    }

    /// The windows of one step of `walk` keys first, `[S, Q]`, where `holder` reads the score
    /// through them; none where it holds the score in its fragments.
    pub fn held_or_open<EA: Float, ES: Float>(
        walk: &Walk,
        #[comptime] holder: ScoreHolder,
    ) -> ComptimeOption<ScoreWindows> {
        match comptime!(holder) {
            ScoreHolder::Window => {
                ComptimeOption::new_Some(ScoreWindows::open::<EA, ES>(walk, comptime!(vec![S, Q])))
            }
            ScoreHolder::Fragments => ComptimeOption::new_None(),
        }
    }

    /// A tile over the box of one step's score, keys first, which the gradients' sums are opened
    /// against: the windows' scores, or, held in fragments, an accumulator over the same box
    /// opened against `key`, which holds no memory.
    pub fn scores_box<EA: Float>(
        windows: &ComptimeOption<ScoreWindows>,
        walk: &Walk,
        key: &Tile,
        #[comptime] score: PlaneFragmentMicrokernel,
    ) -> Tile {
        #[comptime]
        match windows {
            ComptimeOption::Some(windows) => windows.scores.clone(),
            ComptimeOption::None => Tile::stage_accumulator::<EA>(
                walk,
                comptime!(vec![S, Q]),
                key,
                score.tile_instruction(),
            ),
        }
    }

    /// `plane`'s windows, where there are any.
    pub fn held_at(
        windows: &ComptimeOption<ScoreWindows>,
        plane: &Region,
    ) -> ComptimeOption<ScoreWindows> {
        #[comptime]
        match windows {
            ComptimeOption::Some(windows) => ComptimeOption::new_Some(windows.at(plane)),
            ComptimeOption::None => ComptimeOption::new_None(),
        }
    }

    /// Where a step held in fragments leaves the score's gradient cast for a share of the query's
    /// gradient to read transposed: a window of the staged element over one step of `walk`, keys
    /// first, where `holder` keeps the score in its fragments; none where it reads the score
    /// through windows, which leave it in their own cast window.
    pub fn share_window<ES: Float>(
        walk: &Walk,
        #[comptime] holder: ScoreHolder,
    ) -> ComptimeOption<Tile> {
        match comptime!(holder) {
            ScoreHolder::Fragments => ComptimeOption::new_Some(Tile::scratch::<ES>(
                walk,
                comptime!(vec![S, Q]),
                StageStorage::Strided,
            )),
            ScoreHolder::Window => ComptimeOption::new_None(),
        }
    }

    /// The cast window a step leaves the score's gradient in, which a share of the query's
    /// gradient reads transposed: its windows' cast window, or, the score held in fragments, the
    /// window `left` it was left in ([`share_window`](Self::share_window)).
    pub fn cast_of(windows: &ComptimeOption<ScoreWindows>, left: &ComptimeOption<Tile>) -> Tile {
        #[comptime]
        match (windows, left) {
            (ComptimeOption::Some(windows), _) => windows.cast.clone(),
            (ComptimeOption::None, ComptimeOption::Some(left)) => left.clone(),
            (ComptimeOption::None, ComptimeOption::None) => panic!(
                "ScoreWindows::cast_of: a score held in fragments leaves its gradient for a share \
                 only in the window the share opened for it"
            ),
        }
    }

    /// `plane`'s part of `window`, where there is one.
    pub fn window_at(window: &ComptimeOption<Tile>, plane: &Region) -> ComptimeOption<Tile> {
        #[comptime]
        match window {
            ComptimeOption::Some(window) => ComptimeOption::new_Some(window.at(plane)),
            ComptimeOption::None => ComptimeOption::new_None(),
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

    /// `window = lhs · rhs`, summed in `EA` on `product`'s own grid of fragments, placed as
    /// `levels` state it below the walker's planes, then cast into the plane's `window` of shared
    /// memory: where a plane keeps a score, or the gradient against its probabilities, between two
    /// contractions. Each fragment of the grid pairs the operands' fragments by position: the lhs's
    /// rows by steps of the contraction, the rhs's columns by the same steps. The caller meets the
    /// plane before the window is read.
    pub fn contract_into<EA: Float>(
        window: &Tile,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] product: PlaneFragmentMicrokernel,
        #[comptime] levels: Vec<Level>,
    ) {
        // The accumulator, cut to the product's own grid below the planes' level.
        let placed = window.with_levels(1usize, levels);
        let instruction = product.tile_instruction();
        let mut sum = placed.accumulator::<EA>(lhs, rhs, instruction, Semiring::SUM_PROD);
        sum.zero();
        ScoreWindows::contract_fragments(&sum, lhs, rhs, product);
        ScoreWindows::store_cast(window, &sum, product);
    }

    /// `sum`, an accumulator on `product`'s own grid of fragments, cast into the plane's `window`
    /// of shared memory, fragment by fragment. The caller meets the plane before the window is
    /// read.
    pub fn store_cast(window: &Tile, sum: &Tile, #[comptime] product: PlaneFragmentMicrokernel) {
        // The accumulator's fragments and the window are one grid at two depths: the accumulator is
        // placed on the product's own grid, the window sits under the plane's region, and a region
        // only windows a tile at its depth.
        let fragment = comptime!(Level::every(&[
            (product.axes.m, product.instruction.shape()[0].1),
            (product.axes.n, product.instruction.shape()[1].1)
        ]));
        let (window_fragments, sum_fragments) = (window.over(&fragment), sum.over(&fragment));
        #[unroll]
        for at in 0..product.settings.along_m.get() * product.settings.along_n.get() {
            let mut cells = window.at(&window_fragments.region(at));
            cells.copy_cast_from(&sum.at(&sum_fragments.region(at)));
        }
    }

    /// `sum += lhs · rhs` on `product`'s own grid of fragments, `sum` placed on it: each fragment
    /// of the grid pairs the operands' fragments by position, the lhs's rows by steps of the
    /// contraction, the rhs's columns by the same steps. An lhs of manual-mma accumulators is read
    /// in their registers, a run of them along the contraction as one `A`.
    pub fn contract_fragments(
        sum: &Tile,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] product: PlaneFragmentMicrokernel,
    ) {
        // Each operand's fragments, walked over its own box and paired by position.
        let [(_, m), (_, n), (_, k)] = product.instruction.shape();
        let MatmulAxes {
            m: rows_axis,
            n: columns_axis,
            k: depth_axis,
        } = product.axes;
        let rows = product.settings.along_m.get();
        let columns = product.settings.along_n.get();
        let depth = product.settings.depth.get();
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
                        Self::fragment_at(
                            &sum.space(),
                            (rows_axis, row, rows),
                            (columns_axis, column, columns)
                        ),
                        Self::fragment_at(
                            &lhs.space(),
                            (rows_axis, row, rows),
                            (depth_axis, step, depth)
                        ),
                        Self::fragment_at(
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
    }

    /// `lhs · rhs` summed in `EA` on `product`'s own grid of fragments, placed as `levels` state it
    /// below the walker's planes, over `space`, and left in the fragments' registers: where a plane
    /// keeps a score, or the gradient against its probabilities, when the next contraction reads it
    /// cast in place ([`ScoreHolder::Fragments`](crate::attention::plane::ScoreHolder)).
    pub fn contract_held<EA: Float>(
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] space: Space,
        #[comptime] product: PlaneFragmentMicrokernel,
        #[comptime] levels: Vec<Level>,
    ) -> Tile {
        let sum =
            Tile::placed_accumulator::<EA>(space, 1usize, levels, lhs, product.tile_instruction());
        ScoreWindows::contract_fragments(&sum, lhs, rhs, product);
        sum
    }
}

impl ScoreWindows {
    /// The place among a tile's fragments of the one at `first.1` of `first.2` along `first.0` and
    /// `second.1` of `second.2` along `second.0`: a level's regions run along the tile's own axes
    /// in the order its space lays them, the last fastest.
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
}
