use crate::random::Transcendentals;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// One plane's run of the output, each lane drawing from its own generator the lines
/// `lane + i · lanes`: at every draw the plane writes consecutive lines.
///
/// Every device runs it: its lanes share nothing, so a plane of one unit is a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneRandomMicrokernel {
    pub transcendentals: Transcendentals,
    /// Values one lane writes over the plane's run.
    pub values_per_lane: usize,
}

/// How many values a lane writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneRandomMicrokernelSettings {
    pub values_per_lane: usize,
}

impl core::fmt::Display for PlaneRandomMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "values-per-lane={}", self.values_per_lane)
    }
}

impl Microkernel for PlaneRandomMicrokernel {
    type WorkItem = ();
    type Settings = PlaneRandomMicrokernelSettings;
    type Operand = ();
    type CubeSpace = ();

    const DECLINE: &'static str = "the plane random draw runs on every device";

    /// 128 values a lane: enough draws to hide the generator's latency, few enough to unroll.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![PlaneRandomMicrokernelSettings {
            values_per_lane: 128,
        }]
    }

    fn candidates(
        hw: &Hardware,
        _item: &(),
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self {
            transcendentals: Transcendentals::of(hw),
            values_per_lane: settings.values_per_lane,
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
