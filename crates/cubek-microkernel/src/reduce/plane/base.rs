use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row reduced by one plane: its lanes read consecutive lines, each accumulating every `lanes`-th
/// into a partial of its own, and the partials meet across the plane in its instructions.
///
/// It runs where the device commits to a plane width: the partials meet across the plane, so a
/// plane launched wider than the device runs would leave part of the row out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneReduceMicrokernel {
    pub item: ReduceWorkItem,
    /// Units of the plane, as wide as the device commits to.
    pub lanes: usize,
}

/// A plane's microkernel has nothing a measurement settles: its lanes are the device's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneReduceMicrokernelSettings;

impl core::fmt::Display for PlaneReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("whole-plane")
    }
}

impl core::fmt::Display for PlaneReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} {:?} lanes={}",
            self.item.operation, self.item.layout, self.lanes
        )
    }
}

impl Microkernel for PlaneReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = PlaneReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the plane reduce needs a plane width the device commits to";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![PlaneReduceMicrokernelSettings]
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        hw.plane_dim().ok().into_iter().map(|lanes| Self {
            item: *item,
            lanes: lanes as usize,
        })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        self.item.require(vectorization);
    }

    /// Each lane holds its carrier and one line: a lower bound, the registers it keeps live.
    fn estimate(
        &self,
        _item: &ReduceWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let slots = self.item.operation.written_per_row() as u32;
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 2 * slots + 2,
        });
    }
}
