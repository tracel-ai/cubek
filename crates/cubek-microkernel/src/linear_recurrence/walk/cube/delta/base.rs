use crate::linear_recurrence::RecurrenceHeads;
use crate::linear_recurrence::walk::RecurrenceWalkOperand;
use crate::linear_recurrence::walk::cube::{
    CubeWalkMicrokernel, CubeWalkMicrokernelSettings, HeldStateShape,
};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A cube's runs of a head's value columns walked through every position one at a time, held as
/// its [`HeldStateShape`] states: each position decays the state, recalls the key from it and
/// writes the delta.
///
/// It reads no plane instruction and no plane width, only a cube of units and shared memory to
/// meet in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeDeltaWalkMicrokernel {
    pub shape: HeldStateShape,
}

impl core::fmt::Display for CubeDeltaWalkMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.shape.fmt(f)
    }
}

impl Microkernel for CubeDeltaWalkMicrokernel {
    type WorkItem = RecurrenceHeads;
    type Settings = CubeWalkMicrokernelSettings;
    type Operand = RecurrenceWalkOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = HeldStateShape::DECLINE;

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        CubeWalkMicrokernelSettings::listed()
    }

    fn candidates(
        hw: &Hardware,
        item: &RecurrenceHeads,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        HeldStateShape::new(hw, item, settings)
            .map(|shape| Self { shape })
            .into_iter()
    }

    /// Every operand as [`RecurrenceWalkOperand::require`] states it, a run `columns` wide and a
    /// unit's rows.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<RecurrenceWalkOperand>) {
        RecurrenceWalkOperand::require(self.shape.columns(), self.shape.rows(), vectorization);
    }

    fn estimate(
        &self,
        _item: &RecurrenceHeads,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.shape.estimate(estimate);
    }
}

impl CubeWalkMicrokernel for CubeDeltaWalkMicrokernel {
    fn shape(&self) -> HeldStateShape {
        self.shape
    }
}
