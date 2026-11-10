use crate::random::Transcendentals;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// One unit's run of the output, drawn from its own generator line after line.
///
/// Every device runs it: it asks nothing of the plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRandomMicrokernel {
    pub transcendentals: Transcendentals,
}

/// A unit's draws have nothing a measurement settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRandomMicrokernelSettings;

impl core::fmt::Display for UnitRandomMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("one-unit")
    }
}

impl Microkernel for UnitRandomMicrokernel {
    type WorkItem = ();
    type Settings = UnitRandomMicrokernelSettings;
    type Operand = ();
    type CubeSpace = ();

    const DECLINE: &'static str = "the unit random draw runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![UnitRandomMicrokernelSettings]
    }

    fn candidates(
        hw: &Hardware,
        _item: &(),
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self {
            transcendentals: Transcendentals::of(hw),
        })
    }

    /// It reads nothing, and writes whole lines of its run.
    fn vectorize(&self, _: &(), _vectorization: &mut Vectorization<()>) {}

    /// It holds its four generators and the line it writes: a lower bound, the registers it keeps
    /// live per component.
    fn estimate(&self, _item: &(), _: &(), _planes: CubePlanes, estimate: &mut KernelEstimate) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 5,
        });
    }
}
