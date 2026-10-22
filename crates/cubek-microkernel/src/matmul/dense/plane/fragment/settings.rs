use crate::matmul::dense::plane::{PlaneFragmentStepSum, PlaneFragmentTransport};
use core::num::NonZeroUsize;

/// The steps a plane holds along `m` and `n`, and how many steps of `K` it walks per stage, each
/// step the fewest calls of the instruction every load fills whole ([`MatmulStep`](crate::matmul::MatmulStep)). The
/// instruction itself is a candidate, not a setting.
///
/// A warpgroup MMA holds one instruction a plane group, `along_n` times its form's granularity
/// wide: there `along_m` is one, and the steps along `n` size the instruction rather than a grid of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentMicrokernelSettings {
    pub along_m: NonZeroUsize,
    pub along_n: NonZeroUsize,
    pub depth: NonZeroUsize,
    /// The element a software instruction's step contracts in, where its operands admit the
    /// narrow one; a matrix instruction's step is always its own. Wide unless a caller opts in
    /// ([`PlaneFragmentStepSum`]).
    pub step_sum: PlaneFragmentStepSum,
    /// Which transport a matrix instruction is reached through first.
    pub transport: PlaneFragmentTransport,
    /// How deep along `K` each of its fragments is ([`PlaneFragmentDepth`]).
    pub fragment_depth: PlaneFragmentDepth,
}

/// How deep along `K` a plane's fragments are: one instruction step, or every step of a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentDepth {
    /// One instruction step: the plane walks its steps of `K` through a stage one after another,
    /// each a grid of fragments loaded and contracted.
    Instruction,
    /// Every step of a stage: one grid of fragments as deep as the plane's steps, which the
    /// instruction contracts one step after another into the same sum, each step's registers
    /// loaded out of the stage as an instruction-deep fragment's are. A packed operand decodes in
    /// registers as each fragment loads, which is what lets an NVFP4 weight be staged as it is
    /// stored under a 16-bit instruction, or read straight from memory a window of `K` at a time.
    /// Only the manual-mma instruction offers it.
    Stage,
}

impl PlaneFragmentMicrokernelSettings {
    pub const DEFAULT: Self = Self::grid(4, 4, 1);

    /// `along_m` by `along_n` steps a plane, `depth` steps of `K` a stage, wide steps reached
    /// through cubecl's fragment operations.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn grid(along_m: usize, along_n: usize, depth: usize) -> Self {
        Self {
            along_m: NonZeroUsize::new(along_m).unwrap(),
            along_n: NonZeroUsize::new(along_n).unwrap(),
            depth: NonZeroUsize::new(depth).unwrap(),
            step_sum: PlaneFragmentStepSum::Wide,
            transport: PlaneFragmentTransport::Cmma,
            fragment_depth: PlaneFragmentDepth::Instruction,
        }
    }

    /// `along_n` fragments a plane along `n` and one along `m`, each a window `depth` steps of
    /// `K` deep, reached through cubek's transport: the grid a plane holds against a packed weight
    /// it reads straight from memory, decoding it as each fragment loads.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn windowed(along_n: usize, depth: usize) -> Self {
        Self {
            transport: PlaneFragmentTransport::Mma,
            fragment_depth: PlaneFragmentDepth::Stage,
            ..Self::grid(1, along_n, depth)
        }
    }

    /// This grid reaching its matrix instruction through `transport`.
    pub const fn through(self, transport: PlaneFragmentTransport) -> Self {
        Self { transport, ..self }
    }
}

impl core::fmt::Display for PlaneFragmentMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "steps={}x{}x{}", self.along_m, self.along_n, self.depth)?;
        match self.transport {
            PlaneFragmentTransport::Cmma => {}
            PlaneFragmentTransport::Mma => write!(f, " transport=mma")?,
            PlaneFragmentTransport::Wgmma => write!(f, " transport=wgmma")?,
        }
        if self.fragment_depth == PlaneFragmentDepth::Stage {
            write!(f, " fragments_through_stage")?;
        }
        match self.step_sum {
            PlaneFragmentStepSum::Wide => Ok(()),
            PlaneFragmentStepSum::Narrow => write!(f, " narrow"),
        }
    }
}
