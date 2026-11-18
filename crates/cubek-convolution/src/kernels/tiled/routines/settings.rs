use crate::kernels::tiled::{ConvDecline, ConvProblem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::stage::Prefetch;
use cubek_microkernel::matmul::cube::{CubeStagedMicrokernelSettings, MatmulPlaneLayout};
use cubek_microkernel::matmul::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// The width the input and the rhs are read at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedConvVectors {
    One,
    /// The widest the layout, the instruction's steps and the padding admit.
    Widest,
}

impl CubeStagedConvVectors {
    /// The elements one read may bring where a stage tile holds `run` along the innermost axis.
    pub fn cells(self, run: usize) -> usize {
        match self {
            Self::One => 1,
            Self::Widest => run,
        }
    }
}

/// How the output is cut over the planes of a cube and the instructions of a plane, and how the
/// contraction is walked.
///
/// The microkernel's settings hold the grid of fragments a plane holds, the planes along the rows
/// and the columns, and the ring; the rest are the convolution's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedConvSettings {
    pub microkernel: CubeStagedMicrokernelSettings,
    /// Rows of the instruction: the largest the device offers of exactly this many.
    pub rows: NonZeroUsize,
    /// Planes a cube holds along the across axis (OH in a forward), each a row group of its own
    /// over the one stage of the rhs the cube's planes share.
    pub across: NonZeroUsize,
    /// Instruction steps of the contraction one stage holds at most: fewer where the problem has
    /// fewer, no divisor of its walk reaches this far, or the shared memory a cube holds is short.
    pub stage_steps: NonZeroUsize,
    pub vectors: CubeStagedConvVectors,
}

impl CubeStagedConvSettings {
    const STAGE_STEPS: NonZeroUsize = NonZeroUsize::new(4).unwrap();

    /// An instruction of `rows` rows over the grid of `fragments` a plane and `planes`
    /// planes a cube, reading vectors.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn new(
        rows: usize,
        fragments: PlaneFragmentMicrokernelSettings,
        planes: MatmulPlaneLayout,
    ) -> Self {
        Self {
            microkernel: CubeStagedMicrokernelSettings::boxed(fragments, planes),
            rows: NonZeroUsize::new(rows).expect("an instruction has a row"),
            across: NonZeroUsize::MIN,
            stage_steps: Self::STAGE_STEPS,
            vectors: CubeStagedConvVectors::Widest,
        }
    }

    /// These settings with `planes` planes along the across axis, a cube then covering that many
    /// of its positions with each weight stage.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn across(mut self, planes: usize) -> Self {
        self.across = NonZeroUsize::new(planes).expect("a cube holds a plane across");
        self
    }

    /// These settings holding two stages in flight.
    pub const fn double_ring(mut self) -> Self {
        self.microkernel = self.microkernel.ringed(StageRing::DOUBLE);
        self
    }

    /// These settings reading one element at a time.
    pub const fn width_one(mut self) -> Self {
        self.vectors = CubeStagedConvVectors::One;
        self
    }

    /// Planes a cube holds: its grid of them along the rows and the columns, times the across.
    pub fn planes(&self) -> usize {
        self.microkernel.planes.count() * self.across.get()
    }

    /// The instruction of `rows` rows the tile picks for `problem`, with the grid of fragments a
    /// plane holds.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::WgmmaUnserved`], [`ConvDecline::InstructionUnoffered`] and
    /// [`ConvDecline::RowsNotPicked`].
    pub fn plane(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
    ) -> Result<PlaneFragmentMicrokernel, Refusal> {
        let grid = self.microkernel.plane;
        if grid.holder != ComputeScope::Plane {
            return Err(ConvDecline::WgmmaUnserved.into());
        }
        let work_item = problem.work_item();
        let mut candidates = PlaneFragmentMicrokernel::candidates(hw, &work_item, &grid).peekable();
        if candidates.peek().is_none() {
            return Err(ConvDecline::InstructionUnoffered.into());
        }
        candidates
            .find(|plane| {
                let [(_, rows), ..] = plane.instruction.shape();
                rows == self.rows.get()
            })
            .ok_or(ConvDecline::RowsNotPicked.into())
    }
}

impl core::fmt::Display for CubeStagedConvSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (fragments, planes) = (&self.microkernel.plane, &self.microkernel.planes);
        write!(
            f,
            "F{}X{}_P{}X{}_T{}",
            fragments.along_m, fragments.along_n, planes.along_m, planes.along_n, self.rows
        )?;
        match self.across.get() {
            1 => {}
            across => write!(f, "_A{across}")?,
        }
        match self.stage_steps {
            Self::STAGE_STEPS => {}
            stage_steps => write!(f, "_S{stage_steps}")?,
        }
        let ring = self.microkernel.ring;
        match ring.stages_in_flight.get() {
            1 => {}
            stages => write!(f, "_R{stages}")?,
        }
        match ring.prefetch {
            Prefetch::InSlots => {}
            Prefetch::InRegisters => write!(f, "_regs")?,
        }
        match self.vectors {
            CubeStagedConvVectors::One => write!(f, "_W1"),
            CubeStagedConvVectors::Widest => Ok(()),
        }
    }
}
