use super::microkernel::unit_direct_zero_microkernel;
use crate::zero::ZeroOperand;
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A window of an output zeroed by one unit alone. Every device runs it: it asks nothing of the
/// plane.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroMicrokernel {}

#[cube]
impl UnitDirectZeroMicrokernel {
    /// A unit's window `out`, every cell set to zero (`unit_direct_zero_microkernel`).
    pub fn execute(&self, out: &mut Tile) {
        unit_direct_zero_microkernel(out)
    }
}

/// A unit's zeroing has nothing a measurement settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroMicrokernelSettings;

impl core::fmt::Display for UnitDirectZeroMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("one-unit")
    }
}

impl Microkernel for UnitDirectZeroMicrokernel {
    type WorkItem = ();
    type Settings = UnitDirectZeroMicrokernelSettings;
    type Operand = ZeroOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the unit zeroing runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![UnitDirectZeroMicrokernelSettings]
    }

    fn candidates(
        _hw: &Hardware,
        _item: &(),
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self {})
    }

    /// It reads nothing, and writes at whatever width its output is bound at.
    fn vectorize(&self, _: &(), _vectorization: &mut Vectorization<ZeroOperand>) {}

    /// It holds one line of zeros.
    fn estimate(&self, _item: &(), _: &(), _planes: CubePlanes, estimate: &mut KernelEstimate) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 1,
        });
    }
}
