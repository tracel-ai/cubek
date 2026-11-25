use crate::kernels::indexed::m_grouped::{MGroupedMatmulCandidate, MGroupedMatmulTensors};
use crate::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedAsyncCopyMatmulRoutineSettings,
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutine,
    PlaneShuffleMatmulRoutineSettings,
};
use crate::kernels::{ListedFor, Listing, Product};
use core::fmt::Display;
use core::num::NonZeroUsize;
use cubecl::tile::launch::Decline;
use cubek_microkernel::matmul::plane::PlaneShuffleMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::{Routine, first_served_on_floor};

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
///
/// The plane shuffle's by-block lead serves a packed stack whose `K` is no whole number of words
/// of scales, which every other floor declines.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MGroupedFloors {
    pub plane_shuffle: [Listing<MGroupedSettings<PlaneShuffleMatmulRoutineSettings>>; 3],
    pub cube_staged: [Listing<MGroupedSettings<CubeStagedMatmulRoutineSettings>>; 2],
}

impl MGroupedFloors {
    /// Whether some floor listed for the stack's kind prepares `tensors` on `hw`.
    ///
    /// # Errors
    ///
    /// The last floor's decline, where none serves. The first that serves ends the walk.
    pub(crate) fn served(
        &self,
        hw: &Hardware,
        tensors: &MGroupedMatmulTensors,
    ) -> Result<(), Decline> {
        let stack = Product::of([tensors.stack_kind()]);
        let plane_shuffle = Listing::admitted(&self.plane_shuffle, stack)
            .map(|floor| tensors.served_by::<_, _, PlaneShuffleMatmulRoutine>(hw, &floor.settings));
        let cube_staged = Listing::admitted(&self.cube_staged, stack)
            .map(|floor| tensors.served_by::<_, _, CubeStagedMatmulRoutine>(hw, &floor.settings));
        first_served_on_floor(plane_shuffle.chain(cube_staged))
    }

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
            Listing {
                settings: MGroupedSettings::PLANE_SHUFFLE_PACKED_BY_BLOCK,
                kind: ListedFor::Packed,
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

/// The candidates of a grouped product, each routine's list for the stacks each is listed for:
/// those worth racing ([`raced`](Self::raced)), or those every round must be able to fall back on
/// ([`floors`](Self::floors)).
#[derive(Debug, Clone)]
pub struct MGroupedMatmulCandidates {
    /// The plane shuffle's.
    pub plane_shuffle: Vec<MGroupedMatmulCandidate<PlaneShuffleMatmulRoutine>>,
    /// The cube staged routine's.
    pub cube_staged: Vec<MGroupedMatmulCandidate<CubeStagedMatmulRoutine>>,
    /// The copy engine's cube staged routine's, each listed for either kind of stack.
    pub cube_staged_async_copy: Vec<MGroupedMatmulCandidate<CubeStagedAsyncCopyMatmulRoutine>>,
}

impl MGroupedMatmulCandidates {
    /// The settings worth racing on `hw`.
    pub fn raced(hw: &Hardware) -> Self {
        Self {
            plane_shuffle: MGroupedMatmulCandidate::listed(
                PlaneShuffleMatmulRoutine,
                MGroupedSettings::<PlaneShuffleMatmulRoutineSettings>::listed(hw),
            ),
            cube_staged: MGroupedMatmulCandidate::listed(
                CubeStagedMatmulRoutine,
                MGroupedSettings::<CubeStagedMatmulRoutineSettings>::listed(hw),
            ),
            cube_staged_async_copy: MGroupedMatmulCandidate::listed_for_either(
                CubeStagedAsyncCopyMatmulRoutine,
                MGroupedSettings::cube_staged_async_copy(hw),
            ),
        }
    }

    /// The settings every round must be able to fall back on, each for the kind of stack it
    /// serves, none of the copy engine's: a device none of them prepares a product on is refused
    /// before racing ([`MGroupedMatmulTensors::served_on_floor`]).
    pub fn floors() -> Self {
        let floors = MGroupedFloors::LISTED;
        Self {
            plane_shuffle: MGroupedMatmulCandidate::listed(
                PlaneShuffleMatmulRoutine,
                floors.plane_shuffle,
            ),
            cube_staged: MGroupedMatmulCandidate::listed(
                CubeStagedMatmulRoutine,
                floors.cube_staged,
            ),
            cube_staged_async_copy: Vec::new(),
        }
    }
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

    /// The packed lead reading the stack a scale block at a time: what reads a packed stack whose
    /// `K` is no whole number of words of scales.
    pub const PLANE_SHUFFLE_PACKED_BY_BLOCK: Self = Self {
        routine: PlaneShuffleMatmulRoutineSettings::PACKED_BY_BLOCK,
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
        let full = PlaneShuffleMatmulRoutine::settings(hw)
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
    /// own for an indexed packed rhs
    /// ([`indexed_packed_settings`](PlaneShuffleMatmulRoutine::indexed_packed_settings)), over
    /// chunks of the most rows it holds. Not its capped launch
    /// ([`PACKED_PERSISTENT`](PlaneShuffleMatmulRoutineSettings::PACKED_PERSISTENT)): an indexed
    /// launch hands its cubes the chunks beside the columns, which a capped launch declines.
    pub fn plane_shuffle_packed() -> Vec<Self> {
        PlaneShuffleMatmulRoutine::indexed_packed_settings()
            .into_iter()
            .filter(|routine| routine.planes_per_worker.is_none())
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
        Self::over_cube_tiles(CubeStagedMatmulRoutine::settings(hw))
    }

    /// The cube staged routine's settings worth racing on `hw` against a packed stack, each over
    /// chunks of the tile its cubes hold: the dense routine's, then those sized for a group
    /// ([`packed_group_settings`](CubeStagedMatmulRoutine::packed_group_settings)), whose few
    /// dozen rows a dense box pads several times over.
    pub fn cube_staged_packed(hw: &Hardware) -> Vec<Self> {
        let routine = CubeStagedMatmulRoutine;
        let mut settings = routine.packed_settings(hw);
        for candidate in routine.packed_group_settings() {
            if !settings.contains(&candidate) {
                settings.push(candidate);
            }
        }
        Self::over_cube_tiles(settings)
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

impl MGroupedSettings<CubeStagedAsyncCopyMatmulRoutineSettings> {
    /// The copy engine's cube staged routine's settings worth racing on `hw`, each over chunks of
    /// the tile its cubes hold.
    pub fn cube_staged_async_copy(hw: &Hardware) -> Vec<Self> {
        CubeStagedAsyncCopyMatmulRoutine::settings(hw)
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
