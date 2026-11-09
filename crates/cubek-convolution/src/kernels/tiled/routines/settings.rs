use super::{CONV_INSTRUCTION_UNOFFERED, CONV_WGMMA_UNSERVED};
use crate::kernels::tiled::ConvProblem;
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::stage::Prefetch;
use cubek_microkernel::matmul::plane::{
    PlaneFragmentInstruction, PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
    PlaneFragmentTransport,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

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
    /// The largest of exactly this many rows.
    Rows(NonZeroUsize),
}

impl ConvTile {
    fn admits(self, rows: usize) -> bool {
        match self {
            Self::Largest => true,
            Self::Rows(only) => rows == only.get(),
        }
    }
}

impl core::fmt::Display for ConvTile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Largest => Ok(()),
            Self::Rows(rows) => write!(f, "_T{rows}"),
        }
    }
}

/// A count along each of the operation's two free axes, the rows and the columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvAlong {
    pub m: NonZeroUsize,
    pub n: NonZeroUsize,
}

impl ConvAlong {
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn new(m: usize, n: usize) -> Self {
        Self {
            m: NonZeroUsize::new(m).expect("a count along the rows is at least one"),
            n: NonZeroUsize::new(n).expect("a count along the columns is at least one"),
        }
    }
}

impl core::fmt::Display for ConvAlong {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}X{}", self.m, self.n)
    }
}

/// The matrix instruction a plane contracts through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvInstruction {
    /// The transport the planes reach the device's matrix instruction through first.
    pub transport: PlaneFragmentTransport,
    pub tile: ConvTile,
}

impl core::fmt::Display for ConvInstruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.transport {
            PlaneFragmentTransport::Cmma => write!(f, "cmma"),
            PlaneFragmentTransport::Mma => write!(f, "mma"),
            PlaneFragmentTransport::Wgmma => write!(f, "wgmma"),
        }
    }
}

/// How the output is cut over the instructions of a plane and the planes of a cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvGrid {
    /// Instructions a plane holds along the operation's two free axes.
    pub fragments: ConvAlong,
    /// Planes a cube holds along the same two axes.
    pub planes: ConvAlong,
    /// Planes a cube holds along the across axis (OH in a forward), each a row group of its own
    /// over the one stage of the rhs the cube's planes share.
    pub across: NonZeroUsize,
}

impl core::fmt::Display for ConvGrid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F{}_P{}", self.fragments, self.planes)
    }
}

/// How the contraction is walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvWalk {
    /// Instruction steps of the contraction one stage holds, capped by what the problem has.
    pub stage_steps: NonZeroUsize,
    /// Stages a cube holds at once: one fills then contracts, two fill the next while contracting.
    pub ring: StageRing,
}

impl ConvWalk {
    pub const DEFAULT: Self = Self {
        stage_steps: NonZeroUsize::new(4).unwrap(),
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for ConvWalk {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.stage_steps != Self::DEFAULT.stage_steps {
            write!(f, "_S{}", self.stage_steps)?;
        }
        match self.ring.stages_in_flight.get() {
            1 => {}
            stages => write!(f, "_R{stages}")?,
        }
        match self.ring.prefetch {
            Prefetch::InSlots => Ok(()),
            Prefetch::InRegisters => write!(f, "_regs"),
        }
    }
}

/// How the output is cut over the planes of a cube and the instructions of a plane, and how the
/// contraction is walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvSettings {
    pub instruction: ConvInstruction,
    pub grid: ConvGrid,
    pub walk: ConvWalk,
    pub lines: ConvLines,
}

impl ConvSettings {
    /// Cmma over `fragments` instructions a plane and `planes` planes a cube, reading lines.
    pub const fn new(fragments: ConvAlong, planes: ConvAlong) -> Self {
        Self {
            instruction: ConvInstruction {
                transport: PlaneFragmentTransport::Cmma,
                tile: ConvTile::Largest,
            },
            grid: ConvGrid {
                fragments,
                planes,
                across: NonZeroUsize::MIN,
            },
            walk: ConvWalk::DEFAULT,
            lines: ConvLines::Widest,
        }
    }

    /// These settings through the instruction of `rows` rows.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn rows(mut self, rows: usize) -> Self {
        self.instruction.tile =
            ConvTile::Rows(NonZeroUsize::new(rows).expect("an instruction has a row"));
        self
    }

    /// These settings with `planes` planes along the across axis, a cube then covering that many
    /// of its positions with each weight stage.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn across(mut self, planes: usize) -> Self {
        self.grid.across = NonZeroUsize::new(planes).expect("a cube holds a plane across");
        self
    }

    /// These settings through the manual-mma instruction.
    pub const fn mma(mut self) -> Self {
        self.instruction.transport = PlaneFragmentTransport::Mma;
        self
    }

    /// These settings holding two stages in flight.
    pub const fn double_ring(mut self) -> Self {
        self.walk.ring = StageRing::DOUBLE;
        self
    }

    /// These settings holding `steps` instruction steps of the contraction in a stage.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn stage_steps(mut self, steps: usize) -> Self {
        self.walk.stage_steps = NonZeroUsize::new(steps).expect("a stage holds a step");
        self
    }

    /// These settings reading one element at a time.
    pub const fn width_one(mut self) -> Self {
        self.lines = ConvLines::One;
        self
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
        let transport = self.instruction.transport;
        if transport == PlaneFragmentTransport::Wgmma {
            return Err(CONV_WGMMA_UNSERVED);
        }
        let fragments = self.grid.fragments;
        let grid = PlaneFragmentMicrokernelSettings::grid(fragments.m.get(), fragments.n.get(), 1)
            .through(transport);
        PlaneFragmentMicrokernel::candidates(hw, &problem.work_item(), &grid)
            .filter(|plane| {
                let [(_, rows), ..] = plane.instruction.shape();
                self.instruction.tile.admits(rows)
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
        let ConvInstruction { tile, .. } = self.instruction;
        write!(f, "{}_{}{tile}", self.instruction, self.grid)?;
        if self.grid.across.get() > 1 {
            write!(f, "_A{}", self.grid.across)?;
        }
        write!(f, "{}", self.walk)?;
        match self.lines {
            ConvLines::One => write!(f, "_W1"),
            ConvLines::Widest => Ok(()),
        }
    }
}
