use super::CubeStagedConvRoutine;
use crate::kernels::tiled::{ConvProblem, ConvRoutine, ConvTensors};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::plane::{
    PlaneFragmentInstruction, PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
    PlaneFragmentTransport,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// What a setting is declined with where the device offers no instruction of its kind for the
/// problem's elements.
pub const CONV_INSTRUCTION_UNOFFERED: Refusal = Refusal::Declined {
    why: "the device offers no instruction of this kind for the problem's elements",
};

/// What a setting is declined with where it asks for a warpgroup MMA, whose stages only the TMA
/// engine lands and which this convolution's fill never does.
pub const CONV_WGMMA_UNSERVED: Refusal = Refusal::Declined {
    why: "a warpgroup MMA reads its stages as the TMA engine lands them",
};

/// The width the input and the rhs are read at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvLines {
    One,
    /// The widest the layout, the instruction's steps and the padding admit.
    Widest,
}

impl ConvLines {
    /// The elements one read may bring where a stage tile holds `run` along the innermost axis.
    pub fn cells(self, run: usize) -> usize {
        match self {
            Self::One => 1,
            Self::Widest => run,
        }
    }
}

/// Which of the device's matrix instructions of the asked kind a setting takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvTile {
    /// The largest the device offers.
    Largest,
    /// The one of eight rows, which wastes least on a map seven wide.
    EightRows,
}

impl ConvTile {
    fn admits(self, rows: usize) -> bool {
        match self {
            Self::Largest => true,
            Self::EightRows => rows == 8,
        }
    }
}

/// How the output is cut over the planes of a cube and the instructions of a plane, and how the
/// contraction is walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvSettings {
    /// The transport the planes reach the device's matrix instruction through first.
    pub instruction: PlaneFragmentTransport,
    /// Instructions a plane holds along the operation's two free axes.
    pub fragments: (usize, usize),
    /// Planes a cube holds along the same two axes.
    pub planes: (usize, usize),
    /// Instruction steps of the contraction one stage holds, capped by what the problem has.
    pub stage_steps: usize,
    /// Stages a cube holds at once: one fills then contracts, two fill the next while contracting.
    pub ring: StageRing,
    pub lines: ConvLines,
    pub tile: ConvTile,
}

impl ConvSettings {
    pub const fn new(fragments: (usize, usize), planes: (usize, usize), lines: ConvLines) -> Self {
        Self {
            instruction: PlaneFragmentTransport::Cmma,
            fragments,
            planes,
            stage_steps: 4,
            ring: StageRing::SINGLE,
            lines,
            tile: ConvTile::Largest,
        }
    }

    /// These settings through the instruction of eight rows.
    pub const fn eight_rows(self) -> Self {
        Self {
            tile: ConvTile::EightRows,
            ..self
        }
    }

    /// These settings through the manual-mma instruction.
    pub const fn mma(self) -> Self {
        Self {
            instruction: PlaneFragmentTransport::Mma,
            ..self
        }
    }

    /// These settings holding two stages in flight.
    pub const fn double_ring(self) -> Self {
        Self {
            ring: StageRing::DOUBLE,
            ..self
        }
    }

    /// These settings reading one element at a time.
    pub const fn width_one(self) -> Self {
        Self {
            lines: ConvLines::One,
            ..self
        }
    }

    /// The first instruction of the kind these settings ask for that the device offers for
    /// `problem`, with the grid of fragments a plane holds.
    ///
    /// # Errors
    ///
    /// [`CONV_INSTRUCTION_UNOFFERED`], [`CONV_WGMMA_UNSERVED`].
    pub fn plane(
        &self,
        hw: &Hardware,
        problem: &ConvProblem,
    ) -> Result<PlaneFragmentMicrokernel, Refusal> {
        let transport = self.instruction;
        if transport == PlaneFragmentTransport::Wgmma {
            return Err(CONV_WGMMA_UNSERVED);
        }
        let grid = PlaneFragmentMicrokernelSettings::grid(self.fragments.0, self.fragments.1, 1)
            .through(transport);
        PlaneFragmentMicrokernel::candidates(hw, &problem.work_item(), &grid)
            .filter(|plane| {
                let [(_, rows), ..] = plane.instruction.shape();
                self.tile.admits(rows)
            })
            .find(|plane| match transport {
                PlaneFragmentTransport::Cmma => {
                    matches!(plane.instruction, PlaneFragmentInstruction::Cmma { .. })
                }
                PlaneFragmentTransport::Mma => {
                    matches!(plane.instruction, PlaneFragmentInstruction::Mma { .. })
                }
                PlaneFragmentTransport::Wgmma => {
                    matches!(plane.instruction, PlaneFragmentInstruction::Wgmma { .. })
                }
            })
            .ok_or(CONV_INSTRUCTION_UNOFFERED)
    }
}

impl core::fmt::Display for ConvSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let instruction = match self.instruction {
            PlaneFragmentTransport::Cmma => "cmma",
            PlaneFragmentTransport::Mma => "mma",
            PlaneFragmentTransport::Wgmma => "wgmma",
        };
        write!(
            f,
            "{} fragments={}x{} planes={}x{} stage_steps={}",
            instruction,
            self.fragments.0,
            self.fragments.1,
            self.planes.0,
            self.planes.1,
            self.stage_steps
        )?;
        write!(f, "{}", self.ring)?;
        match self.lines {
            ConvLines::One => write!(f, " width=1"),
            ConvLines::Widest => Ok(()),
        }?;
        match self.tile {
            ConvTile::Largest => Ok(()),
            ConvTile::EightRows => write!(f, " rows=8"),
        }
    }
}

/// The routines a caller names, each at a setting or at the first it lists that serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvRoutineKind {
    /// [`CubeStagedConvRoutine`].
    CubeStaged,
}

impl core::fmt::Display for ConvRoutineKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeStaged => write!(f, "cube_staged"),
        }
    }
}

impl ConvRoutineKind {
    /// `tensors` launched on `hw` by this routine at `settings`, or at the first it lists that
    /// serves where none is named.
    ///
    /// # Errors
    ///
    /// Why the routine declines the convolution on `hw`.
    pub fn launch_on(
        &self,
        hw: &Hardware,
        tensors: ConvTensors,
        settings: Option<&ConvSettings>,
    ) -> Result<(), Refusal> {
        match self {
            Self::CubeStaged => CubeStagedConvRoutine.launch_on(hw, tensors, settings),
        }
    }
}
