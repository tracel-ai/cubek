use core::fmt::Display;
use core::num::NonZeroUsize;
use cubek_microkernel::matmul::dense::plane::PlaneShuffleMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

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

    /// The plane shuffle's settings worth racing on `hw`: the dense routine's own for a float
    /// rhs, then those for a packed one, over chunks of the most rows it holds; then its
    /// default over shorter chunks, which pad a small group less and read a large one's slab more
    /// often. Each setting declines the stack it does not read.
    pub fn plane_shuffle(hw: &Hardware) -> Vec<Self> {
        let chunk =
            |rows| MGroupedChunk::Rows(NonZeroUsize::new(rows).expect("a chunk holds a row"));
        let most = PlaneShuffleMicrokernel::MAX_ROWS;
        let float = PlaneShuffleMatmulRoutine.settings(hw);
        let packed: Vec<_> = PlaneShuffleMatmulRoutine
            .packed_settings()
            .into_iter()
            .filter(|settings| !float.contains(settings))
            .collect();
        let full = float.into_iter().chain(packed).map(|routine| Self {
            routine,
            chunk: chunk(most),
        });
        let shorter = [most / 2, most / 4].map(|rows| Self {
            chunk: chunk(rows),
            ..Self::PLANE_SHUFFLE
        });
        full.chain(shorter).collect()
    }
}

impl MGroupedSettings<CubeStagedMatmulRoutineSettings> {
    /// The cube staged routine's default over chunks of the tile its cubes hold.
    pub const CUBE_STAGED: Self = Self {
        routine: CubeStagedMatmulRoutineSettings::DEFAULT,
        chunk: MGroupedChunk::CubeTile,
    };

    /// The cube staged routine's settings worth racing on `hw`, each over chunks of the tile its
    /// cubes hold.
    pub fn cube_staged(hw: &Hardware) -> Vec<Self> {
        CubeStagedMatmulRoutine
            .settings(hw)
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
