use super::stage_storage;
use crate::attention::plane::ScoreHolder;
use crate::attention::{Q, S};
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

    /// The windows of one step of `walk` along `axes`, where `holder` reads the score through
    /// them; none where it holds the score in its fragments.
    pub fn held_or_open<EA: Float, ES: Float>(
        walk: &Walk,
        #[comptime] axes: Vec<Axis>,
        #[comptime] holder: ScoreHolder,
    ) -> ComptimeOption<ScoreWindows> {
        match holder {
            ScoreHolder::Window => {
                ComptimeOption::new_Some(ScoreWindows::open::<EA, ES>(walk, axes))
            }
            ScoreHolder::Fragments => ComptimeOption::new_None(),
        }
    }

    /// What each step contracts a gradient's sum with, over the box of one step's score laid out
    /// along `axes`, which the sum is opened against: the windows' cast to the operands' element,
    /// or, held in fragments, an accumulator over the same box opened against `against`, which
    /// holds no memory.
    pub fn cast_box<EA: Float>(
        windows: &ComptimeOption<ScoreWindows>,
        walk: &Walk,
        #[comptime] axes: Vec<Axis>,
        against: &Tile,
    ) -> Tile {
        #[comptime]
        match windows {
            ComptimeOption::Some(windows) => windows.cast.clone(),
            ComptimeOption::None => Tile::stage_accumulator::<EA>(walk, axes, against),
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
    /// first, where `holder` keeps the score in its fragments, laid out as the walk's stages are
    /// for the loads that read it ([`stage_storage`]); none where it reads the score through
    /// windows, which leave it in their own cast window.
    pub fn share_window<ES: Float>(
        walk: &Walk,
        #[comptime] holder: ScoreHolder,
        #[comptime] score: PlaneFragmentMicrokernel,
    ) -> ComptimeOption<Tile> {
        match holder {
            ScoreHolder::Fragments => ComptimeOption::new_Some(Tile::scratch::<ES>(
                walk,
                comptime!(vec![S, Q]),
                comptime!(stage_storage(holder, score)),
            )),
            ScoreHolder::Window => ComptimeOption::new_None(),
        }
    }

    /// The cast window a step read through windows leaves the score's gradient in, which a share
    /// of the query's gradient reads transposed.
    pub fn cast_of(windows: &ComptimeOption<ScoreWindows>) -> Tile {
        #[comptime]
        match windows {
            ComptimeOption::Some(windows) => windows.cast.clone(),
            ComptimeOption::None => panic!(
                "ScoreWindows::cast_of: a score held in fragments leaves its gradient in the \
                 window its share opened, which the share adds by rows"
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

    /// `window = lhs · rhs`, summed in `EA` on the plane's tile placed as `levels` state it below
    /// the walker's planes, then drained into the plane's `window` of shared memory: where a plane
    /// keeps a score, or the gradient against its probabilities, between two contractions. The
    /// tile cuts the plane's tile into the calls of the instruction it picks and walks the
    /// contraction. The caller meets the plane before the window is read.
    pub fn contract_into<EA: Float>(
        window: &Tile,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] levels: Vec<Level>,
    ) {
        // The window, the accumulator and both operands on one grid below the planes' level: the
        // tile's walk of the contraction steps all three at that depth.
        let placed = window.with_levels(1usize, levels.clone());
        let lhs = lhs.with_levels(1usize, levels.clone());
        let rhs = rhs.with_levels(1usize, levels);
        let mut sum = placed.accumulator::<EA>(&lhs, &rhs, Semiring::SUM_PROD);
        sum.zero();
        sum.mma(&lhs, &rhs);
        sum.drained_into(&placed);
    }

    /// `lhs · rhs` summed in `EA` on the plane's tile, placed as `levels` state it below the
    /// walker's planes, over `space`, and left in the fragments' registers: where a plane keeps a
    /// score, or the gradient against its probabilities, when the next contraction reads it cast
    /// in place ([`ScoreHolder::Fragments`](crate::attention::plane::ScoreHolder)). The tile cuts
    /// the sum into cells the next contraction reads in runs.
    pub fn contract_held<EA: Float>(
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] space: Space,
        #[comptime] levels: Vec<Level>,
    ) -> Tile {
        let lhs = lhs.with_levels(1usize, levels.clone());
        let rhs = rhs.with_levels(1usize, levels.clone());
        let mut sum = Tile::placed_accumulator::<EA>(space, 1usize, levels, &lhs);
        sum.mma(&lhs, &rhs);
        sum
    }
}
