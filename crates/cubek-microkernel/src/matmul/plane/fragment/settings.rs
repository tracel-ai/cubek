use core::num::NonZeroUsize;
use cubecl::tile::space::ComputeScope;

/// The shape of the tile a plane holds along `m` and `n`, and how deep along `K` it walks per
/// stage, in units: steps of the device's matrix unit for the product's elements, its largest
/// square instruction
/// ([`PlaneFragmentMicrokernel::matrix_unit`](super::PlaneFragmentMicrokernel::matrix_unit)), each
/// the fewest calls of it every load fills whole ([`MatmulStep`](crate::matmul::MatmulStep)). So a
/// setting is one tile shape in elements on a device, whichever instruction cuts it. The
/// instruction is not a setting: the tile layer picks it for the product
/// ([`Instruction::for_accumulator`](cubecl::tile::Instruction::for_accumulator)).
///
/// A resolved [`PlaneFragmentMicrokernel`](super::PlaneFragmentMicrokernel) states the same
/// fields in its own instruction's calls: the tile cut into fragments.
///
/// A plane group MMA holds one instruction a plane group, `along_n` times its form's granularity
/// wide: there `along_m` is one, and the steps along `n` size the instruction rather than a grid of
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentMicrokernelSettings {
    pub along_m: NonZeroUsize,
    pub along_n: NonZeroUsize,
    pub depth: NonZeroUsize,
    /// Who holds each of the plane's tiles: the plane itself, or a plane group the tiles are dealt
    /// to whole. A plane group ([`ComputeScope::MMA_PLANE_GROUP`]) holds its tile in the plane
    /// group MMA, which the tile layer picks for it: a routine of its own, whose stages only the
    /// TMA engine lands.
    pub holder: ComputeScope,
    /// How deep along `K` each of its fragments is ([`PlaneFragmentDepth`]).
    pub fragment_depth: PlaneFragmentDepth,
    /// The axis the instruction's rows, and its accumulator's, lie along
    /// ([`PlaneFragmentRowsAlong`]).
    pub rows_along: PlaneFragmentRowsAlong,
}

/// The axis a plane's instruction lays its rows along: which factor fills its `A`, the other its
/// `B`. Along `m`, the lhs is `A`, as the product names it. Along `n`, the rhs is: a packed weight
/// decoded into `A` writes each value straight into the register `A` reads it from, and the
/// activation in `B` loads a lane's run of `k` as one row's run, where an `A` takes two rows
/// interleaved a register at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneFragmentRowsAlong {
    M,
    N,
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
    /// registers as each fragment loads, which is what lets an NVFP4 input be staged as it is
    /// stored under a 16-bit instruction, or read straight from memory a window of `K` at a time.
    /// Only the manual-mma instruction over plain operands offers it
    /// ([`contracts_deeper_fragments`](super::PlaneFragmentInstruction::contracts_deeper_fragments)).
    Stage,
}

impl PlaneFragmentMicrokernelSettings {
    pub const DEFAULT: Self = Self::grid(4, 4, 1);

    /// `along_m` by `along_n` units a plane, `depth` units of `K` a stage.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn grid(along_m: usize, along_n: usize, depth: usize) -> Self {
        Self {
            along_m: NonZeroUsize::new(along_m).unwrap(),
            along_n: NonZeroUsize::new(along_n).unwrap(),
            depth: NonZeroUsize::new(depth).unwrap(),
            holder: ComputeScope::Plane,
            fragment_depth: PlaneFragmentDepth::Instruction,
            rows_along: PlaneFragmentRowsAlong::M,
        }
    }

    /// `along_n` fragments a plane along `n` and one along `m`, each a window `depth` steps of `K`
    /// deep: the grid a plane holds against a packed rhs it reads straight from memory, decoding
    /// it as each fragment loads, which only the instruction contracting fragments a stage deep
    /// does ([`PlaneFragmentDepth::Stage`]).
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn windowed(along_n: usize, depth: usize) -> Self {
        Self {
            fragment_depth: PlaneFragmentDepth::Stage,
            ..Self::grid(1, along_n, depth)
        }
    }

    /// This grid with its tiles held by `holder`.
    pub const fn held_by(self, holder: ComputeScope) -> Self {
        Self { holder, ..self }
    }

    /// This grid with its instruction's rows along `rows_along`.
    pub const fn rows_along(self, rows_along: PlaneFragmentRowsAlong) -> Self {
        Self { rows_along, ..self }
    }
}

impl core::fmt::Display for PlaneFragmentMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "units={}x{}x{}", self.along_m, self.along_n, self.depth)?;
        match self.holder {
            ComputeScope::Plane => {}
            ComputeScope::PlaneGroup { planes } => write!(f, " plane_groups={planes}")?,
            other => write!(f, " holder={other:?}")?,
        }
        if self.fragment_depth == PlaneFragmentDepth::Stage {
            write!(f, " fragments_through_stage")?;
        }
        if self.rows_along == PlaneFragmentRowsAlong::N {
            write!(f, " rows_along=n")?;
        }
        Ok(())
    }
}
