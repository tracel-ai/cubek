use super::settings::{CubeWalkMicrokernelSettings, MAX_HELD_LINES_PER_UNIT};
use crate::linear_recurrence::RecurrenceHeads;
use cubek_std::Hardware;
use cubek_std::microkernel::{ArithmeticInstruction, KernelEstimate, UnitWork};

/// How a cube holds a run's state: `columns` value columns a run, cut over its key rows into one
/// tile of `rows` rows for each of `units` units, whose partial sums over the rows meet in the
/// cube's shared memory.
///
/// It holds what was derived, not the settings it was derived from: two settings that derive one
/// launch are one candidate, which a round compiles and times once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeldStateShape {
    /// Units sharing a run: a power of two, which the tile engine's units level needs. Units
    /// past the head's last key row hold nothing.
    units: usize,
    /// Key rows each unit holds of the run's state, a multiple of `columns`.
    rows: usize,
    /// Value columns a run holds: a fact of the heads, and the widest line the walk reads.
    columns: usize,
}

impl HeldStateShape {
    /// Why a walk declines a head [`new`](Self::new) holds no state of.
    pub const DECLINE: &'static str =
        "a cube's power-of-two units cannot hold the key rows of one run in masked or whole tiles";

    /// The units holding the cells the settings target: the fewest, a power of two, whose tiles
    /// of at most that many rows hold the head, with the rows that spreads it over them. None
    /// where a cube holds no such count, or where the last tile would leave a unit wholly past the
    /// head without masking it: the tile engine masks tiles past the head only where the rows do
    /// not divide it.
    pub fn new(
        hw: &Hardware,
        heads: &RecurrenceHeads,
        settings: &CubeWalkMicrokernelSettings,
    ) -> Option<Self> {
        let columns = heads.columns();
        let key_features = heads.key_features;
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
        (fits && masked).then_some(Self {
            units,
            rows,
            columns,
        })
    }

    pub fn units(&self) -> usize {
        self.units
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Vector lines of the state a unit holds at lines of `width` columns.
    pub fn held_lines(&self, width: usize) -> usize {
        self.rows * self.columns / width
    }

    /// The work each unit holds: its rows of the state beside its partial sums, computed with the
    /// device's vector arithmetic. The reads are its routine's to count.
    pub fn estimate(&self, estimate: &mut KernelEstimate) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: (self.rows * self.columns + 3 * self.columns) as u32,
        });
    }

    /// Bytes of shared memory a cube of `runs` runs meets in: two slices, each holding every
    /// unit's partial for the run's columns.
    pub fn shared_memory(&self, runs: usize) -> usize {
        2 * runs * self.units * self.columns * size_of::<f32>()
    }
}

impl core::fmt::Display for HeldStateShape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "units={} rows={} columns={}",
            self.units, self.rows, self.columns
        )
    }
}
