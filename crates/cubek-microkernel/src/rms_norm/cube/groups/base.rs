use crate::rms_norm::{HeldRowShape, RmsNormOperand, RmsNormWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A cube's rows, each owned by a group of its units that need not be a plane: each unit holds its
/// lines in registers, and a group's partials sum through the cube's shared memory, halving each
/// step, the cube meeting between steps.
///
/// Under a plain store it asks nothing of the hardware — no plane instruction, no width the device
/// commits to — so it runs everywhere, and a group of one is a unit owning its row outright, which
/// sums nothing; a packed store's encoding meets through planes
/// ([`RowStore::runs_on`](crate::row_store::RowStore::runs_on)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeGroupsRmsNormMicrokernel {
    /// Units sharing a row: the settings' target snapped to a power of two, grown to as many as
    /// keep each unit within [`HeldRowShape::MAX_LINES_PER_UNIT`] lines of the row at `width`, and
    /// no larger than the lines the row holds then, so no unit holds nothing while the others hold
    /// several, nor than a cube holds.
    pub units: usize,
    /// The dim the row spans. An extent the kernel compiles, the exception
    /// [`RmsNormPartitioning::kernel`](crate::rms_norm::RmsNormPartitioning::kernel)
    /// argues: the units size their registers by it, and the table's key holds it exactly.
    pub dim: usize,
    /// The width the operands bind at ([`RmsNormWorkItem::width`]), which the units are sized at.
    pub width: usize,
}

/// How many units a row is shared by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeGroupsRmsNormMicrokernelSettings {
    /// A target, not a demand ([`CubeGroupsRmsNormMicrokernel::units`]).
    pub units: NonZeroUsize,
}

impl CubeGroupsRmsNormMicrokernelSettings {
    pub const DEFAULT: Self = Self::new(256);

    /// Groups of `units` units.
    ///
    /// # Panics
    ///
    /// On no units.
    pub const fn new(units: usize) -> Self {
        Self {
            units: NonZeroUsize::new(units).unwrap(),
        }
    }
}

impl core::fmt::Display for CubeGroupsRmsNormMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "units={}", self.units)
    }
}

impl CubeGroupsRmsNormMicrokernel {
    /// How a unit's share of the row lies at the width the operands bind at.
    pub fn held_row(&self) -> HeldRowShape {
        HeldRowShape {
            lines: self.dim.div_ceil(self.width),
            units: self.units,
        }
    }

    /// Bytes of shared memory one group's partials meet in: one `f32` a unit, and none where a unit
    /// owns its row.
    pub fn shared_memory(&self) -> usize {
        match self.units {
            1 => 0,
            units => units * size_of::<f32>(),
        }
    }
}

/// Its units.
impl core::fmt::Display for CubeGroupsRmsNormMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "units={}", self.units)
    }
}

impl Microkernel for CubeGroupsRmsNormMicrokernel {
    type WorkItem = RmsNormWorkItem;
    type Settings = CubeGroupsRmsNormMicrokernelSettings;
    type Operand = RmsNormOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube's groups ask nothing of the hardware";

    /// The default, then groups of fewer units, down to a unit a row: what a prompt's many rows
    /// want, each short enough for few units to hold.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [256, 64, 16, 1].map(Self::Settings::new).into()
    }

    /// The group the settings target, grown to keep each unit's share within its registers and
    /// bounded by the row's lines and a cube's units ([`units`](Self::units)), each a power of two.
    /// Every device runs it.
    fn candidates(
        hw: &Hardware,
        item: &RmsNormWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let power_below = |count: usize| 1usize << count.max(1).ilog2();
        let lines = item.lines(hw);
        let needed = lines
            .div_ceil(HeldRowShape::MAX_LINES_PER_UNIT)
            .next_power_of_two();
        let ceiling = power_below(lines).min(power_below(
            hw.properties.hardware.max_units_per_cube as usize,
        ));
        let units = power_below(settings.units.get()).max(needed).min(ceiling);
        core::iter::once(Self {
            units,
            dim: item.dim,
            width: item.width(hw),
        })
    }

    /// Every operand along the whole row ([`RmsNormOperand::require_along_run`]).
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<RmsNormOperand>) {
        RmsNormOperand::require_along_run(self.dim, vectorization);
    }

    /// Its units hold their share of the row in registers ([`HeldRowShape::work`]). The reads are
    /// its routine's to count, which places its groups.
    fn estimate(
        &self,
        item: &RmsNormWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(self.held_row().work(self.width, item.element));
    }
}
