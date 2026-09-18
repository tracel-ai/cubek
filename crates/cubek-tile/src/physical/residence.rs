//! Where a stage lives, how it lays its cells out ([`StageStorage`]) and what it leaves between
//! its fragment rows ([`Pitch`]).

use crate::{Axis, Space, StageForm};

/// Bytes of one shared-memory bank line: the banks a phase of a fragment read crosses.
const BANK_LINE_BYTES: usize = 128;

/// Rows one phase of a fragment read covers, and so the chunks a bank line is cut into for them.
const PHASE_ROWS: usize = 8;

/// The chunk a pitch is counted in: a bank line over the rows one phase reads.
const PITCH_CHUNK_BYTES: usize = BANK_LINE_BYTES / PHASE_ROWS;

/// Where a stage lives and how it lays its cells out: shared memory, storage-tiled at a stated
/// block (one contiguous block per fragment, what a cmma transaction wants) or in plain strided
/// rows; or the plane's own lanes, which is no buffer at all. Stated by the kernel where it
/// allocates the stage ([`Ring::smem`](crate::Ring::smem)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    /// Grouped into `block`-sized tiles, the fragment the instruction reads: one edge per axis of
    /// the operation, of which an operand takes its own.
    Tiled {
        block: Vec<(Axis, usize)>,
        pitch: Pitch,
    },
    Strided,
    /// Not shared memory at all: the plane holds the lines in its lanes, one to a lane, and
    /// shares them by shuffle ([`Lanes`](crate::Lanes)).
    Lanes,
}

/// The pitch between a tiled stage's fragment rows: what one row starts after the row above it.
///
/// A fragment is read a phase at a time, and a phase reads [`PHASE_ROWS`] rows across the banks of
/// one line. Rows that start a whole *odd* number of chunks apart land those reads on that many
/// distinct chunks, one each; rows laid end to end at an even count of chunks collide, two or more
/// rows to a chunk, and the phase is replayed once per collision.
///
/// Serialized because a caller's setting rides a persisted autotune key, so the
/// value a winner was measured at has to come back.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
)]
pub enum Pitch {
    /// Rows lie end to end: the fragment is its own rows and nothing between them.
    #[default]
    Dense,
    /// Rows start the least odd number of chunks apart that holds one, so a phase reads each of
    /// its rows out of a chunk of its own.
    Padded,
}

impl Pitch {
    /// The bytes one fragment row starts after the row above it, given the bytes it holds.
    ///
    /// Never a stated number: [`Padded`](Pitch::Padded) rounds the row up to whole chunks and
    /// then to an odd count of them, so a 32-byte row pitches at 48 and a 64-byte one at 80,
    /// while a row that is already one chunk is left alone.
    pub fn of(&self, row_bytes: usize) -> usize {
        match self {
            Pitch::Dense => row_bytes,
            // `| 1` is the least odd count at or above the row's own, which is the whole rule:
            // an odd count is coprime with the chunks of a bank line, so consecutive rows walk
            // all of them before any repeats.
            Pitch::Padded => (row_bytes.div_ceil(PITCH_CHUNK_BYTES).max(1) | 1) * PITCH_CHUNK_BYTES,
        }
    }
}

impl StageStorage {
    /// What this storage leaves between a fragment's rows. A plain strided stage groups no
    /// fragment, so its rows are its own and nothing sits between them.
    pub(crate) fn pitch(&self) -> Pitch {
        match self {
            StageStorage::Tiled { pitch, .. } => *pitch,
            StageStorage::Strided => Pitch::Dense,
            StageStorage::Lanes => {
                panic!("StageStorage::Lanes: the plane's lanes are not shared memory")
            }
        }
    }

    /// Lines a stage over `space` reserves under this storage, served `vector_size` wide at
    /// `elem_size` bytes an element — the buffer's own size, the pitch's padding included.
    ///
    /// The one place that arithmetic lives, so a host-side budget for a stage and the stage the
    /// kernel allocates cannot disagree about what it costs.
    pub fn cells(&self, space: &Space, vector_size: usize, elem_size: usize) -> usize {
        StageForm::dense(space, vector_size, self.clone(), elem_size).cells()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule the pitch exists for: a phase's rows land on chunks of their own.
    ///
    /// Stated as the property rather than as the three numbers it produces, since what a row is
    /// worth padding to is arithmetic about the banks and not a value anyone measured.
    #[test]
    fn a_padded_row_starts_an_odd_number_of_chunks_after_the_one_above_it() {
        for row_bytes in [16usize, 32, 48, 64, 96, 128] {
            let pitch = Pitch::Padded.of(row_bytes);
            assert!(pitch >= row_bytes, "a pitch holds its own row");
            assert_eq!(pitch % PITCH_CHUNK_BYTES, 0, "a pitch is whole chunks");
            let chunks = pitch / PITCH_CHUNK_BYTES;
            assert_eq!(
                chunks % 2,
                1,
                "{row_bytes} bytes pitched to {chunks} chunks"
            );
            // The *least* such pitch: one chunk less either does not hold the row or is even.
            let under = pitch - PITCH_CHUNK_BYTES;
            assert!(under < row_bytes || (under / PITCH_CHUNK_BYTES).is_multiple_of(2));
        }
    }

    /// A dense pitch is the row itself, which is what keeps a stage that states none byte for
    /// byte the stage it was before there was one to state.
    #[test]
    fn a_dense_row_is_its_own_pitch() {
        for row_bytes in [16usize, 32, 48, 64] {
            assert_eq!(Pitch::Dense.of(row_bytes), row_bytes);
        }
    }
}
