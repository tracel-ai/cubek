use core::fmt::Display;
use core::num::NonZeroUsize;
use cubek_microkernel::matmul::plane::PlaneShuffleMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

use crate::kernels::tune::{ListedFor, Listing};
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutine,
    PlaneShuffleMatmulRoutineSettings,
};

/// A dense routine's setting a grouped launch runs at, and the chunks of rows it pads each group
/// to: the launch is the dense product of those chunks, one a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MGroupedSettings<S> {
    pub routine: S,
    pub chunk: MGroupedChunk,
}

/// The rows of a grouped launch's chunks, each group padded to a whole number of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MGroupedChunk {
    /// These many: what a routine holding every row of its batch whole holds of a group at once.
    Rows(NonZeroUsize),
    /// The rows of the tile one cube holds, as the routine prepares the grouped rows as one
    /// product: a routine that tiles its rows then holds one group in each cube.
    CubeTile,
}

/// The settings every round must be able to fall back on, each for the kind of stack it serves.
/// Each declines where another runs: the plane shuffle where the device commits to no plane
/// width, its default against a packed stack, which its lead against a packed rhs reads, and the
/// cube staged routine on a CPU runtime, which holds fewer units a cube than its planes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MGroupedFloors {
    pub plane_shuffle: [Listing<MGroupedSettings<PlaneShuffleMatmulRoutineSettings>>; 2],
    pub cube_staged: [Listing<MGroupedSettings<CubeStagedMatmulRoutineSettings>>; 2],
}

impl MGroupedFloors {
    pub(crate) const LISTED: Self = Self {
        plane_shuffle: [
            Listing {
                settings: MGroupedSettings::PLANE_SHUFFLE,
                kind: ListedFor::Either,
            },
            Listing {
                settings: MGroupedSettings::PLANE_SHUFFLE_PACKED,
                kind: ListedFor::Either,
            },
        ],
        cube_staged: [
            Listing {
                settings: MGroupedSettings::CUBE_STAGED,
                kind: ListedFor::Float,
            },
            Listing {
                settings: MGroupedSettings::CUBE_STAGED_PACKED,
                kind: ListedFor::Packed,
            },
        ],
    };
}

impl MGroupedSettings<PlaneShuffleMatmulRoutineSettings> {
    /// The plane shuffle's default over chunks of the most rows it holds: a group's chunk reads
    /// its slab once.
    pub const PLANE_SHUFFLE: Self = Self {
        routine: PlaneShuffleMatmulRoutineSettings::DEFAULT,
        chunk: MGroupedChunk::Rows(NonZeroUsize::new(PlaneShuffleMicrokernel::MAX_ROWS).unwrap()),
    };

    /// The plane shuffle's lead against a packed rhs over the same chunks: what reads a packed
    /// stack where the default declines it.
    pub const PLANE_SHUFFLE_PACKED: Self = Self {
        routine: PlaneShuffleMatmulRoutineSettings::PACKED,
        ..Self::PLANE_SHUFFLE
    };

    /// Every plane shuffle setting worth racing on `hw`, for the stacks each is listed for. Each
    /// declines a stack it cannot read, so the float ones race on a packed stack too: some read
    /// one faster than the packed lead does.
    pub(crate) fn listed(
        hw: &Hardware,
    ) -> Vec<Listing<MGroupedSettings<PlaneShuffleMatmulRoutineSettings>>> {
        Listing::float_then_packed(Self::plane_shuffle(hw), Self::plane_shuffle_packed())
    }

    /// The plane shuffle's settings worth racing on `hw` against a float stack: the dense
    /// routine's own over chunks of the most rows it holds; then its default over shorter chunks,
    /// which pad a small group less and read a large one's slab more often.
    pub fn plane_shuffle(hw: &Hardware) -> Vec<Self> {
        let most = PlaneShuffleMicrokernel::MAX_ROWS;
        let full = PlaneShuffleMatmulRoutine
            .settings(hw)
            .into_iter()
            .map(|routine| Self {
                routine,
                ..Self::PLANE_SHUFFLE
            });
        let shorter = [most / 2, most / 4].map(|rows| Self {
            chunk: MGroupedChunk::Rows(NonZeroUsize::new(rows).expect("a chunk holds a row")),
            ..Self::PLANE_SHUFFLE
        });
        full.chain(shorter).collect()
    }

    /// The plane shuffle's settings worth racing against a packed stack: the dense routine's
    /// own for a packed rhs, over chunks of the most rows it holds.
    pub fn plane_shuffle_packed() -> Vec<Self> {
        PlaneShuffleMatmulRoutine
            .packed_settings()
            .into_iter()
            .map(|routine| Self {
                routine,
                ..Self::PLANE_SHUFFLE
            })
            .collect()
    }
}

impl MGroupedSettings<CubeStagedMatmulRoutineSettings> {
    /// The cube staged routine's default over chunks of the tile its cubes hold.
    pub const CUBE_STAGED: Self = Self {
        routine: CubeStagedMatmulRoutineSettings::DEFAULT,
        chunk: MGroupedChunk::CubeTile,
    };

    /// The cube staged routine's lead against a packed stack, over the same chunks.
    pub const CUBE_STAGED_PACKED: Self = Self {
        routine: CubeStagedMatmulRoutineSettings::PACKED,
        ..Self::CUBE_STAGED
    };

    /// Every cube staged setting worth racing on `hw`, for the stacks each is listed for.
    pub(crate) fn listed(
        hw: &Hardware,
    ) -> Vec<Listing<MGroupedSettings<CubeStagedMatmulRoutineSettings>>> {
        Listing::merged(Self::cube_staged(hw), Self::cube_staged_packed(hw))
    }

    /// The cube staged routine's settings worth racing on `hw` against a float stack, each over
    /// chunks of the tile its cubes hold.
    pub fn cube_staged(hw: &Hardware) -> Vec<Self> {
        Self::over_cube_tiles(CubeStagedMatmulRoutine.settings(hw))
    }

    /// The cube staged routine's settings worth racing on `hw` against a packed stack, each over
    /// chunks of the tile its cubes hold.
    pub fn cube_staged_packed(hw: &Hardware) -> Vec<Self> {
        Self::over_cube_tiles(CubeStagedMatmulRoutine.packed_settings(hw))
    }

    fn over_cube_tiles(settings: Vec<CubeStagedMatmulRoutineSettings>) -> Vec<Self> {
        settings
            .into_iter()
            .map(|routine| Self {
                routine,
                chunk: MGroupedChunk::CubeTile,
            })
            .collect()
    }
}

impl<S: Display> Display for MGroupedSettings<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.chunk {
            MGroupedChunk::Rows(rows) => write!(f, "{} chunk_rows={rows}", self.routine),
            MGroupedChunk::CubeTile => write!(f, "{} chunk_rows=cube", self.routine),
        }
    }
}
