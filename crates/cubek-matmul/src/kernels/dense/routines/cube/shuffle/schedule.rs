use super::CubeShuffleMatmulRoutineSettings;
use core::num::NonZeroUsize;
use cubecl::tile::layout::Run;
use cubek_microkernel::matmul::cube::{CubeShuffleMicrokernelSettings, CubeShufflePlaneSettings};
use cubek_microkernel::matmul::plane::{PlaneShuffleKSplit, PlaneShuffleMicrokernelSettings};

/// How the cube shuffle schedules the loading of a packed rhs whose `K` its planes split: who
/// reads which of the data, and when, at each level of the machine, finest first.
///
/// The data comes with its layout, which the schedule does not choose: its addresses are digits,
/// a tile of tile order's strip, a block of the tile, a row of it, a word of the block. A
/// schedule hands each digit to a level of the machine: what one lane's load takes
/// ([`by_block`](Self::by_block)), the plane's units along `K` beside the rest of the plane along
/// `N` ([`units_along_k`](Self::units_along_k)), the columns a unit holds at once
/// ([`columns`](Self::columns)), the planes of a cube along `K`, which take its steps in turns
/// ([`planes`](Self::planes)), and the cubes
/// ([`cubes_per_worker`](Self::cubes_per_worker)); time takes what is left. Every schedule reads
/// every byte once, so each is correct; they differ in the runs of memory the loads in flight at
/// once fall on, which is what the device's memory serves fast or slow. Which is fastest is the
/// device's and the layout's, so they are raced ([`space`](Self::space)), not chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoadingSchedule {
    /// Whether a lane's load is one block of one row, its scale read alone, or the load the rhs
    /// is stored in, a word of scales a turn.
    pub by_block: bool,
    /// The plane's units along `K`, the rest of the plane along `N`: the whole plane, or as many as
    /// read one memory transaction or one stored tile whole, as the rhs is stored.
    pub units_along_k: PlaneShuffleKSplit,
    /// Columns one unit holds at once.
    pub columns: NonZeroUsize,
    /// Planes of a cube along `K`.
    pub planes: NonZeroUsize,
    /// Cubes a compute unit, each walking the boxes of columns a launch apart; a cube a box where
    /// `None`.
    pub cubes_per_worker: Option<NonZeroUsize>,
}

impl LoadingSchedule {
    /// The decode-step split of `K`: a packed rhs read a block at a time, a unit eight columns,
    /// every plane of the cube one turn of `K`, up to thirty-two planes, on four cubes a compute
    /// unit: each load a plane issues is one run of a column in rows, and every unit holds the lhs
    /// for its one block of `K`. On a Radeon 8060S it reads NVFP4 Qwen3-8B
    /// projections in rows at one row at 98–100% of a row sum of the weight.
    pub const SPLIT_K: Self = Self {
        by_block: true,
        units_along_k: PlaneShuffleKSplit::WholePlane,
        columns: NonZeroUsize::new(8).unwrap(),
        planes: NonZeroUsize::new(32).unwrap(),
        cubes_per_worker: Some(NonZeroUsize::new(4).unwrap()),
    };

    /// [`SPLIT_K`](Self::SPLIT_K) with as many units along `K` as read one memory transaction
    /// whole, the rest a column each along `N`: the fastest split of a weight in rows on a Radeon 8060S from one to
    /// eight rows on most projections.
    pub const ONE_TRANSACTION: Self = Self {
        units_along_k: PlaneShuffleKSplit::Run(Run::Transaction),
        columns: NonZeroUsize::MIN,
        ..Self::SPLIT_K
    };

    /// [`SPLIT_K`](Self::SPLIT_K) with as many units along `K` as read one stored tile, two
    /// columns a unit and the cubes uncapped: a cube's turn reads
    /// one run of whole tiles. The fastest split of a weight in tile order at one row on a Radeon
    /// 8060S (block by block) and an L4 (row by row).
    pub const ONE_TILE: Self = Self {
        units_along_k: PlaneShuffleKSplit::Run(Run::StoredTile),
        columns: NonZeroUsize::new(2).unwrap(),
        cubes_per_worker: None,
        ..Self::SPLIT_K
    };

    /// The schedules measured fastest somewhere, which the few-rows autotune races against a
    /// packed rhs: the rest of [`space`](Self::space) would multiply every tune for points no
    /// projection has been measured to win on.
    pub const MEASURED: [Self; 3] = [Self::SPLIT_K, Self::ONE_TRANSACTION, Self::ONE_TILE];

    /// Every schedule the cube shuffle can run: each load, each placement of a plane's units,
    /// one to eight columns a unit, four to thirty-two planes, and the cubes
    /// a box each or one to four a compute unit. Schedules a layout or device cannot size decline
    /// when prepared. What a bench explores to find the next point of
    /// [`MEASURED`](Self::MEASURED).
    pub fn space() -> impl Iterator<Item = Self> {
        let count = |value: usize| NonZeroUsize::new(value).expect("a count of at least one");
        let mut space = Vec::new();
        for by_block in [true, false] {
            for units_along_k in [
                PlaneShuffleKSplit::WholePlane,
                PlaneShuffleKSplit::Run(Run::Transaction),
                PlaneShuffleKSplit::Run(Run::StoredTile),
            ] {
                for columns in [1, 2, 4, 8].map(count) {
                    for planes in [4, 8, 16, 32].map(count) {
                        for cubes_per_worker in [None, Some(1), Some(2), Some(4)] {
                            space.push(Self {
                                by_block,
                                units_along_k,
                                columns,
                                planes,
                                cubes_per_worker: cubes_per_worker.map(count),
                            });
                        }
                    }
                }
            }
        }
        space.into_iter()
    }

    /// The routine settings that run this schedule.
    pub const fn settings(self) -> CubeShuffleMatmulRoutineSettings {
        let plane = PlaneShuffleMicrokernelSettings {
            k_split: self.units_along_k,
            ..PlaneShuffleMicrokernelSettings::stepping(self.columns.get(), None)
        };
        CubeShuffleMatmulRoutineSettings {
            microkernel: CubeShuffleMicrokernelSettings {
                plane: CubeShufflePlaneSettings::Shuffle(plane),
                planes_along_k: self.planes,
                by_block: self.by_block,
            },
            cubes_per_worker: self.cubes_per_worker,
        }
    }
}

impl core::fmt::Display for LoadingSchedule {
    /// The label of the settings it runs, which the autotune keys its results by.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The measured schedules are points of the space, and the space names each schedule once.
    #[test]
    fn the_measured_schedules_lie_in_the_space() {
        let space: Vec<_> = LoadingSchedule::space().collect();
        let distinct: HashSet<_> = space.iter().collect();
        assert_eq!(distinct.len(), space.len());
        for schedule in LoadingSchedule::MEASURED {
            assert!(distinct.contains(&schedule), "{schedule}");
        }
    }

    /// The measured schedules keep the labels the autotune has stored their results under.
    #[test]
    fn the_measured_schedules_keep_their_labels() {
        let labels = LoadingSchedule::MEASURED.map(|schedule| schedule.to_string());
        assert_eq!(
            labels,
            [
                "cube-shuffle-matmul block=8x1 k_split=whole-plane planes_along_k=32 by_block \
                 cubes_per_worker=4",
                "cube-shuffle-matmul block=1x1 k_split=one-transaction planes_along_k=32 \
                 by_block cubes_per_worker=4",
                "cube-shuffle-matmul block=2x1 k_split=one-tile planes_along_k=32 by_block",
            ]
        );
    }
}
