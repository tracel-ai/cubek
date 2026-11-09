use crate::linear_recurrence::RecurrenceHeads;
use crate::linear_recurrence::walk::RecurrenceWalkOperand;
use crate::linear_recurrence::walk::cube::{CubeWalkMicrokernelSettings, HeldStateShape};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

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

    /// The default, then half its cells.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![Self::Settings::DEFAULT, Self::Settings::new(32)]
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

    /// Each unit holds its rows of the state beside its partial sums, computed with the device's
    /// vector arithmetic; the reads are its routine's to count.
    fn estimate(
        &self,
        _item: &RecurrenceHeads,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let (rows, columns) = (self.shape.rows(), self.shape.columns());
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: (rows * columns + 3 * columns) as u32,
        });
    }
}
