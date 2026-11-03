use crate::gated_delta::GatedDeltaHeads;
use crate::gated_delta::walk::GatedDeltaWalkOperand;
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// The most vector lines of a row or a state a unit holds in registers: past it the registers, not
/// the data, bound the kernel, and its unrolled walk takes longer to compile than the launch it
/// serves takes to run.
pub const MAX_HELD_LINES_PER_UNIT: usize = 16;

/// A cube's runs of a head's value columns, `columns` of them a run, walked through every
/// position one at a time by `units` units: the partitioning cuts a run's key rows into one tile
/// of `rows` rows for each unit, which the unit holds of the run's state in its registers, and the
/// units' partial sums over the key rows meet in the cube's shared memory.
///
/// It reads no plane instruction and no plane width, only a cube of units and shared memory to
/// meet in. It holds what it derived, not the settings it derived it from: two settings that
/// derive one launch are one candidate, which a round compiles and times once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeGatedDeltaWalkMicrokernel {
    /// Units sharing a run: a power of two, which the tile engine's units level needs. Units
    /// past the head's last key row hold nothing.
    pub units: usize,
    /// Key rows each unit holds of the run's state, a multiple of `columns`.
    pub rows: usize,
    /// Value columns a run holds: a fact of the heads, and the widest line the walk reads.
    pub columns: usize,
}

/// How much of a run's state one unit holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeGatedDeltaWalkMicrokernelSettings {
    /// A target, not a demand: the `f32` cells of the state one unit holds in registers. Fewer
    /// cells is more units a run, and wider meetings.
    pub cells: NonZeroUsize,
}

impl CubeGatedDeltaWalkMicrokernelSettings {
    /// 64 cells a unit: a 128-row head read four columns wide in runs of eight units.
    pub const DEFAULT: Self = Self::new(64);
    /// The fewest cells whose lines stay within [`MAX_HELD_LINES_PER_UNIT`] at any width.
    pub const FLOOR: Self = Self::new(MAX_HELD_LINES_PER_UNIT);

    pub const fn new(cells: usize) -> Self {
        Self {
            cells: NonZeroUsize::new(cells).unwrap(),
        }
    }

    /// These settings with no more than `cells` cells a unit.
    pub fn capped_at(self, cells: usize) -> Self {
        Self::new(self.cells.get().min(cells))
    }
}

impl core::fmt::Display for CubeGatedDeltaWalkMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cells={}", self.cells)
    }
}

impl CubeGatedDeltaWalkMicrokernel {
    /// Vector lines of the state a unit holds at lines of `width` columns.
    pub fn held_lines(&self, width: usize) -> usize {
        self.rows * self.columns / width
    }

    /// Bytes of shared memory a cube of `runs` runs meets in: two slices, one for each sum of a
    /// position, each holding every unit's partial for the run's columns.
    pub fn shared_memory(&self, runs: usize) -> usize {
        2 * runs * self.units * self.columns * size_of::<f32>()
    }
}

impl core::fmt::Display for CubeGatedDeltaWalkMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "units={} rows={} columns={}",
            self.units, self.rows, self.columns
        )
    }
}

impl Microkernel for CubeGatedDeltaWalkMicrokernel {
    type WorkItem = GatedDeltaHeads;
    type Settings = CubeGatedDeltaWalkMicrokernelSettings;
    type Operand = GatedDeltaWalkOperand;
    type CubeSpace = ();

    const DECLINE: &'static str =
        "a cube's power-of-two units cannot hold the key rows of one run in masked or whole tiles";

    /// The default, then half its cells.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![Self::Settings::DEFAULT, Self::Settings::new(32)]
    }

    /// The units holding the cells the settings target: the fewest, a power of two, whose tiles
    /// of at most that many rows hold the head, with the rows that spreads it over them. None
    /// where a cube holds no such count, or where the last tile would leave a unit wholly past the
    /// head without masking it: the tile engine masks tiles past the head only where the rows do
    /// not divide it.
    fn candidates(
        hw: &Hardware,
        item: &GatedDeltaHeads,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let columns = item.columns();
        let key_features = item.key_features;
        let target = settings
            .cells
            .get()
            .div_ceil(columns)
            .next_multiple_of(columns)
            .min(MAX_HELD_LINES_PER_UNIT)
            .min(key_features);
        let units = key_features.div_ceil(target).next_power_of_two();
        let rows = key_features.div_ceil(units).next_multiple_of(columns);
        let hardware = &hw.properties.hardware;
        let fits = units <= hardware.max_units_per_cube as usize
            && units <= hardware.max_cube_dim.0 as usize;
        let masked = units * rows == key_features || !key_features.is_multiple_of(rows);
        (fits && masked)
            .then_some(Self {
                units,
                rows,
                columns,
            })
            .into_iter()
    }

    /// Every operand as [`GatedDeltaWalkOperand::require`] states it, a run `columns` wide and a
    /// unit's rows.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<GatedDeltaWalkOperand>) {
        GatedDeltaWalkOperand::require(self.columns, self.rows, vectorization);
    }

    /// Each unit holds its rows of the state beside its partial sums, computed with the device's
    /// vector arithmetic; the reads are its routine's to count.
    fn estimate(
        &self,
        _item: &GatedDeltaHeads,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: (self.rows * self.columns + 3 * self.columns) as u32,
        });
    }
}
