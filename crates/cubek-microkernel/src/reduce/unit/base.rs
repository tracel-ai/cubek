use super::microkernel::unit_direct_reduce_microkernel;
use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row reduced by one unit alone: it reads every line of the row and accumulates each as it lands.
/// Across rows, its lines hold a row a lane, so one unit reduces `width` rows at once.
///
/// Every device runs it: it asks nothing of the plane.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
}

impl UnitDirectReduceMicrokernel {
    /// A unit reducing `item`.
    pub fn new(item: ReduceWorkItem) -> Self {
        Self { item }
    }

    /// Its `item`.
    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }
}

#[cube]
impl UnitDirectReduceMicrokernel {
    /// One unit's row of `input` reduced into `out` and `indices` ([`unit_direct_reduce_microkernel`]).
    pub fn execute<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
        &self,
        input: &Tile,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        unit_direct_reduce_microkernel::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            comptime!(UnitDirectReduceMicrokernel { item: self.item }),
        )
    }
}

/// A unit's microkernel has nothing a measurement settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceMicrokernelSettings;

impl core::fmt::Display for UnitDirectReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("one-unit")
    }
}

impl core::fmt::Display for UnitDirectReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} {:?}", self.item.operation, self.item.layout)
    }
}

impl Microkernel for UnitDirectReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = UnitDirectReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the unit reduce runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![UnitDirectReduceMicrokernelSettings]
    }

    fn candidates(
        _hw: &Hardware,
        item: &ReduceWorkItem,
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self { item: *item })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        self.item.require(vectorization);
    }

    /// It holds its carrier and one line: a lower bound, the registers it keeps live.
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
