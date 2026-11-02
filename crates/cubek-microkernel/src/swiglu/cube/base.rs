use crate::swiglu::{SwigluOperand, SwigluWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A cube's runs, one row's run each, dealt across a group of its units: unit `u` of a group takes
/// lines `u + i · units` of the run, so consecutive units read consecutive lines, and no unit waits
/// on another.
///
/// A run is whole lines of its row and divides it, so no tile overhangs the row: a tile that did
/// would be masked an element at a time. Under a plain store it asks nothing of the hardware — no
/// plane instruction, no width the device commits to — so it runs everywhere; a packed store's
/// encoding meets through planes ([`RowStore::runs_on`](crate::row_store::RowStore::runs_on)).
///
/// It holds what it derived, not the settings it derived it from: two settings that derive one
/// launch are one candidate, which a round compiles and times once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeSwigluMicrokernel {
    /// Lines of the row one run spans ([`candidates`](Self::candidates)). An extent the kernel
    /// compiles, as the width a [`SwigluPartitioning`] compiles in is: the units walk their share
    /// of it, and the autotune key holds the width exactly.
    ///
    /// [`SwigluPartitioning`]: crate::swiglu::SwigluPartitioning
    pub run_lines: usize,
    /// Units sharing a run: the settings' target snapped to a power of two, no larger than a cube
    /// holds, nor than the run has lines, so no unit holds nothing while the others hold several.
    pub units: usize,
    /// The width the operands bind at ([`SwigluWorkItem::width`]), which the run is counted in.
    pub width: usize,
}

/// How many units share a run, and how many lines each of them takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeSwigluMicrokernelSettings {
    /// A target, not a demand ([`CubeSwigluMicrokernel::units`]).
    pub units: NonZeroUsize,
    /// A target, not a demand: the lines a unit takes where its run holds that many for each of
    /// its units ([`CubeSwigluMicrokernel::run_lines`]).
    pub lines: NonZeroUsize,
}

impl CubeSwigluMicrokernelSettings {
    /// A run of 256 units, four lines each: a decode step's row of a model's `d_ff` in a few
    /// cubes, each unit with several loads in flight.
    pub const DEFAULT: Self = Self {
        units: NonZeroUsize::new(256).unwrap(),
        lines: NonZeroUsize::new(4).unwrap(),
    };
    /// 256 units, a line each: the same row over four times the cubes.
    pub const ONE_LINE: Self = Self {
        units: NonZeroUsize::new(256).unwrap(),
        lines: NonZeroUsize::new(1).unwrap(),
    };
    /// 64 units, a line each: a decode step's single row over the most cubes.
    pub const SPREAD: Self = Self {
        units: NonZeroUsize::new(64).unwrap(),
        lines: NonZeroUsize::new(1).unwrap(),
    };
    /// 64 units, four lines each: a prompt's rows, several groups a cube.
    pub const SHORT_RUNS: Self = Self {
        units: NonZeroUsize::new(64).unwrap(),
        lines: NonZeroUsize::new(4).unwrap(),
    };
    /// 64 units, sixteen lines each: a prompt's rows, a unit carrying more of its row.
    pub const LONG_RUNS: Self = Self {
        units: NonZeroUsize::new(64).unwrap(),
        lines: NonZeroUsize::new(16).unwrap(),
    };

    /// The most lines one unit unrolls: past it an unrolled walk takes longer to compile than the
    /// launch it serves takes to run, and a longer share is walked in a loop.
    pub const MAX_UNROLLED_LINES: usize = 16;
}

impl core::fmt::Display for CubeSwigluMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "units={} lines={}", self.units, self.lines)
    }
}

impl CubeSwigluMicrokernel {
    /// Elements of the row one run spans.
    pub fn run(&self) -> usize {
        self.run_lines * self.width
    }

    /// Lines one unit takes: every `units`-th of the run, the last units' share one short where
    /// the units do not divide it.
    pub fn lines_per_unit(&self) -> usize {
        self.run_lines.div_ceil(self.units)
    }

    /// Whether a unit's walk of its lines is unrolled: within
    /// [`MAX_UNROLLED_LINES`](CubeSwigluMicrokernelSettings::MAX_UNROLLED_LINES).
    pub fn unrolled(&self) -> bool {
        self.lines_per_unit() <= CubeSwigluMicrokernelSettings::MAX_UNROLLED_LINES
    }
}

/// Its run and its units.
impl core::fmt::Display for CubeSwigluMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "run={} units={}", self.run_lines, self.units)
    }
}

impl Microkernel for CubeSwigluMicrokernel {
    type WorkItem = SwigluWorkItem;
    type Settings = CubeSwigluMicrokernelSettings;
    type Operand = SwigluOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube's runs ask nothing of the hardware";

    /// The default, then the same group a line each, then groups of 64 units: of a line each,
    /// which spreads a decode step's single row over the most cubes, and of more lines, what a
    /// prompt's many rows want.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::ONE_LINE,
            Self::Settings::SPREAD,
            Self::Settings::SHORT_RUNS,
            Self::Settings::LONG_RUNS,
        ]
    }

    /// The group the settings target, a power of two no larger than a cube holds, and the longest
    /// run dividing the row that it holds at the lines the settings target, and that the store
    /// holds, or the whole row where no such run fills the group, a prime count of lines among
    /// them; the group shrinks to the run where the run is shorter. Every device runs it.
    fn candidates(
        hw: &Hardware,
        item: &SwigluWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let power_below = |count: usize| 1usize << count.max(1).ilog2();
        let row_lines = item.lines(hw);
        let units = power_below(settings.units.get()).min(power_below(
            hw.properties.hardware.max_units_per_cube as usize,
        ));
        let lines = settings.lines.get().min(Self::Settings::MAX_UNROLLED_LINES);
        let width = item.width(hw);
        let divisor = (1..=(units * lines).min(row_lines))
            .rev()
            .find(|&run| row_lines.is_multiple_of(run) && item.store.holds_cube_run(run * width))
            .unwrap_or(row_lines);
        let run_lines = match divisor < units.min(row_lines) {
            true => row_lines,
            false => divisor,
        };
        core::iter::once(Self {
            run_lines,
            units: units.min(power_below(run_lines)),
            width,
        })
    }

    /// Every operand along its run ([`SwigluOperand::require_along_run`]).
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<SwigluOperand>) {
        SwigluOperand::require_along_run(self.run(), vectorization);
    }

    /// Each unit holds a line of either half at a time beside its result, computed with the
    /// device's vector arithmetic; the reads are its routine's to count.
    fn estimate(
        &self,
        item: &SwigluWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let bytes = 3 * self.width * item.element.size();
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: bytes.div_ceil(4) as u32,
        });
    }
}
