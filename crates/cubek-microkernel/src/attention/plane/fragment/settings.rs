use crate::matmul::dense::plane::{PlaneFragmentInstruction, PlaneFragmentTransport};
use core::num::NonZeroUsize;
use cubecl::tile::Scratch;

/// The instructions a plane's box holds along the queries and along a stage of keys, and how
/// much of its output a correction bounces through shared memory at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionMicrokernelSettings {
    pub seq_q: NonZeroUsize,
    pub seq_kv: NonZeroUsize,
    /// One fragment at a time, three `sync_plane` each, or the whole grid at once, three a stage
    /// for a window as large as the plane's output.
    pub scratch: Scratch,
    /// Which transport a matrix instruction is reached through first: cubek's own keeps the score
    /// in its fragments ([`ScoreHolder::Fragments`](super::ScoreHolder::Fragments)), cubecl's
    /// reads it through the plane's windows.
    pub transport: PlaneFragmentTransport,
    /// The shape of cmma instruction tried first, where the plane reads the score through its
    /// windows; the other where the device offers none.
    pub shape: InstructionShape,
}

/// A matrix instruction's shape: as long as it is wide and deep, or any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstructionShape {
    Square,
    Rectangular,
}

impl InstructionShape {
    /// Whether `instruction` is of this shape.
    pub fn holds(self, instruction: PlaneFragmentInstruction) -> bool {
        let [(_, m), (_, n), (_, k)] = instruction.shape();
        (m == n && n == k) == (self == Self::Square)
    }
}

impl PlaneFragmentAttentionMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        seq_q: NonZeroUsize::new(2).unwrap(),
        seq_kv: NonZeroUsize::new(2).unwrap(),
        scratch: Scratch::OneTile,
        transport: PlaneFragmentTransport::Mma,
        shape: InstructionShape::Square,
    };

    /// The default with every fragment of the grid bounced at once.
    pub const WHOLE_GRID: Self = Self {
        scratch: Scratch::WholeGrid,
        ..Self::DEFAULT
    };

    /// The default with `rows` fragments of rows and `keys` of keys.
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

    /// The default reading the score through the windows of the vendor's transport, on an
    /// instruction of `shape`.
    pub const fn through_windows(shape: InstructionShape) -> Self {
        Self {
            transport: PlaneFragmentTransport::Cmma,
            shape,
            ..Self::DEFAULT
        }
    }
}

impl core::fmt::Display for PlaneFragmentAttentionMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fragments={}x{}", self.seq_q, self.seq_kv)?;
        match self.scratch {
            Scratch::WholeGrid => write!(f, " scratch=whole")?,
            Scratch::OneTile | Scratch::None => {}
        }
        match self.transport {
            PlaneFragmentTransport::Mma => {}
            PlaneFragmentTransport::Cmma => write!(f, " transport=cmma")?,
            PlaneFragmentTransport::Wgmma => write!(f, " transport=wgmma")?,
        }
        match self.shape {
            InstructionShape::Square => Ok(()),
            InstructionShape::Rectangular => write!(f, " shape=rectangular"),
        }
    }
}
