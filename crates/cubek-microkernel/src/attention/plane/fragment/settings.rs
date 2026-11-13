use core::num::NonZeroUsize;
use cubecl::tile::Scratch;

/// The shape of the tile a plane holds along the queries and along a stage of keys, in units of the
/// device's matrix unit for the operands
/// ([`matrix_unit`](crate::matmul::plane::PlaneFragmentMicrokernel::matrix_unit)), how much of its
/// output a correction bounces through shared memory at once, and where it keeps the score. The
/// instruction is not a setting: the tile layer picks it where the score is held, and it cuts the
/// same tile whichever it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionMicrokernelSettings {
    pub seq_q: NonZeroUsize,
    pub seq_kv: NonZeroUsize,
    /// One fragment at a time, three `sync_plane` each, or the whole grid at once, three a stage
    /// for a window as large as the plane's output.
    pub scratch: Scratch,
    /// Where the plane keeps the score between its two multiplications.
    pub score: ScoreHolder,
}

/// Where a plane keeps a stage's score between its two multiplications. The tile layer picks the
/// instruction from it: a score opened in windows takes the accumulator's pick, one kept in its
/// fragments the pick for a sum contracted again where it lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreHolder {
    /// The plane's windows of shared memory: the score drained into one, the softmax read there,
    /// the probabilities cast into the other and loaded again as fragments. What a cmma fragment,
    /// whose cells lie where the instruction keeps them, is read through.
    Window,
    /// The score's own manual-mma fragments: the softmax reads them where the plane's units hold
    /// them, and the value contracts them, cast, in the same registers. Each unit keeps the state
    /// of the rows it holds cells of, so the plane's state merges with no other holder's. Only a
    /// device whose instruction lays its `A` as it lays its accumulator keeps it so.
    Fragments,
}

impl PlaneFragmentAttentionMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        seq_q: NonZeroUsize::new(2).unwrap(),
        seq_kv: NonZeroUsize::new(2).unwrap(),
        scratch: Scratch::OneTile,
        score: ScoreHolder::Fragments,
    };

    /// The default with every fragment of the grid bounced at once.
    pub const WHOLE_GRID: Self = Self {
        scratch: Scratch::WholeGrid,
        ..Self::DEFAULT
    };

    /// The default reading the score through its windows.
    pub const THROUGH_WINDOWS: Self = Self::DEFAULT.held_in(ScoreHolder::Window);

    /// The default with `rows` units of rows and `keys` of keys.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn fragments(rows: usize, keys: usize) -> Self {
        Self {
            seq_q: NonZeroUsize::new(rows).unwrap(),
            seq_kv: NonZeroUsize::new(keys).unwrap(),
            ..Self::DEFAULT
        }
    }

    /// These settings keeping the score in `score`.
    pub const fn held_in(self, score: ScoreHolder) -> Self {
        Self { score, ..self }
    }
}

impl core::fmt::Display for PlaneFragmentAttentionMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fragments={}x{}", self.seq_q, self.seq_kv)?;
        match self.scratch {
            Scratch::WholeGrid => write!(f, " scratch=whole")?,
            Scratch::OneTile | Scratch::None => {}
        }
        match self.score {
            ScoreHolder::Fragments => Ok(()),
            ScoreHolder::Window => write!(f, " score=window"),
        }
    }
}
