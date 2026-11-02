use crate::rms_norm::{HeldRowShape, RmsNormOperand, RmsNormWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A cube's rows, each owned by a group of its units that need not be a plane: each unit holds its
/// lines in registers, and a group's partials sum as [`GroupSum`] says, by plane instructions where
/// the device commits to a plane width, else through the cube's shared memory.
///
/// Under a plain store it asks nothing of the hardware — it sums through shared memory where the
/// device has no plane instruction or no width it commits to — so it runs everywhere, and a group of one is a unit owning its row outright, which
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
    /// argues: the units size their registers by it, and the autotune key holds it exactly.
    pub dim: usize,
    /// The width the operands bind at ([`RmsNormWorkItem::width`]), which the units are sized at.
    pub width: usize,
    /// How a group's units sum their partials.
    pub sum: GroupSum,
}

/// How a group's units sum their partials into the row's sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupSum {
    /// A group of one: its unit's partial is the row's.
    Alone,
    /// A group within one plane: shuffles between its lanes, no shared memory and no meeting.
    Shuffles,
    /// A group of whole planes, `lanes` units each: each plane sums its lanes, its first lane
    /// leaves the sum in the plane's slot, and every unit adds the group's slots.
    Planes { lanes: usize },
    /// No plane width to count on: one slot a unit, summed in halves, the cube meeting between
    /// every step.
    Halving,
}

impl GroupSum {
    /// How a group of `units`, a power of two, sums on `hw`.
    pub fn of(hw: &Hardware, units: usize) -> Self {
        match (units, hw.plane_dim()) {
            (1, _) => Self::Alone,
            (units, Ok(lanes)) if units <= lanes as usize => Self::Shuffles,
            (units, Ok(lanes)) if units.is_multiple_of(lanes as usize) => Self::Planes {
                lanes: lanes as usize,
            },
            _ => Self::Halving,
        }
    }

    /// `f32` slots one group's partials meet in.
    pub fn slots(&self, units: usize) -> usize {
        match self {
            Self::Alone | Self::Shuffles => 0,
            Self::Planes { lanes } => units / lanes,
            Self::Halving => units,
        }
    }
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

    /// Bytes of shared memory one group's partials meet in ([`GroupSum::slots`]).
    pub fn shared_memory(&self) -> usize {
        self.sum.slots(self.units) * size_of::<f32>()
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
    /// bounded by the row's lines and a cube's units ([`units`](Self::units)), each a power of two,
    /// summing as the device allows ([`GroupSum::of`]). Every device runs it.
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
            sum: GroupSum::of(hw, units),
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::devices;

    /// A group sums by plane instructions only where the device commits to a plane width: within
    /// one plane by shuffles, across whole planes through one slot a plane, and through one slot
    /// a unit everywhere else.
    #[test]
    fn a_group_sums_by_planes_where_the_device_commits_to_a_width() {
        let m2 = devices::m2_pro();
        let lanes = m2.plane_dim().unwrap() as usize;
        assert_eq!(GroupSum::of(&m2, 1), GroupSum::Alone);
        assert_eq!(GroupSum::of(&m2, lanes / 2), GroupSum::Shuffles);
        assert_eq!(GroupSum::of(&m2, lanes), GroupSum::Shuffles);
        assert_eq!(GroupSum::of(&m2, 8 * lanes), GroupSum::Planes { lanes });
        assert_eq!(GroupSum::Planes { lanes }.slots(8 * lanes), 8);
        assert_eq!(GroupSum::Shuffles.slots(lanes), 0);
        for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
            assert_eq!(GroupSum::of(&hw, 64), GroupSum::Halving);
            assert_eq!(GroupSum::Halving.slots(64), 64);
        }
    }
}
