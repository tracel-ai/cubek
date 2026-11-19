use super::microkernel::unit_direct_reduce_microkernel;
use crate::reduce::{ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOTS};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A row reduced by one unit alone: it reads every line of the row and accumulates each as it lands.
/// Across rows, its lines hold a row a lane, so one unit reduces `width` rows at once.
///
/// Every device runs it: it asks nothing of the plane.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// As [`UnitDirectReduceMicrokernelSettings::lane_slots`](crate::reduce::unit::UnitDirectReduceMicrokernelSettings::lane_slots).
    #[cube(comptime)]
    lane_slots: usize,
}

impl UnitDirectReduceMicrokernel {
    /// A unit reducing `item`, keeping `lane_slots` top-k slots across a line.
    pub fn new(item: ReduceWorkItem, lane_slots: usize) -> Self {
        Self { item, lane_slots }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    /// The widest line its carrier keeps apart at its lane slots, where the carrier bounds it.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slots(self.lane_slots)
    }
}

#[cube]
impl UnitDirectReduceMicrokernel {
    /// One unit's row of `input` reduced into `out` and `indices` (`unit_direct_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where its item writes the indices beside the values and `indices` is none.
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
            comptime!(UnitDirectReduceMicrokernel {
                item: self.item,
                lane_slots: self.lane_slots
            }),
        )
    }
}

/// What a measurement settles for a unit's microkernel: its top-k lane slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceMicrokernelSettings {
    /// Top-k slots a thread keeps across a line's lanes, which bound a top-k's line width
    /// ([`ReduceOperation::widest_line_at_lane_slots`](crate::reduce::ReduceOperation::widest_line_at_lane_slots)).
    pub lane_slots: usize,
}

impl core::fmt::Display for UnitDirectReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lane_slots={}", self.lane_slots)
    }
}

impl core::fmt::Display for UnitDirectReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} {:?}", self.item.operation, self.item.lines_along)
    }
}

impl Microkernel for UnitDirectReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = UnitDirectReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the unit reduce runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        TOP_K_LANE_SLOTS
            .map(|lane_slots| UnitDirectReduceMicrokernelSettings { lane_slots })
            .to_vec()
    }

    fn candidates(
        _hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self {
            item: *item,
            lane_slots: settings.lane_slots,
        })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        vectorization.require(self.item.requirement());
    }

    /// It holds its carrier and one line: a lower bound, the registers it keeps live.
    fn estimate(
        &self,
        _item: &ReduceWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(self.item.unit_work());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::{ReduceOperation, TOP_K_LANE_SLOTS};
    use cubek_std::devices;

    /// Only a top-k bounds the line, to its lane slots spread over `k`.
    #[test]
    fn its_widest_line_is_its_carriers() {
        const LANE_SLOTS: usize = 32;
        const K: usize = 4;
        let sum = UnitDirectReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_SLOTS,
        );
        let top_k = UnitDirectReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::TopK { k: K }),
            LANE_SLOTS,
        );
        assert_eq!(sum.widest_line(), None);
        assert_eq!(top_k.widest_line(), Some(LANE_SLOTS / K));
    }

    /// One setting a lane-slot count, and a candidate on a device lacking every feature.
    #[test]
    fn it_runs_on_a_device_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let settings = UnitDirectReduceMicrokernel::settings(&hw);
        assert_eq!(settings.len(), TOP_K_LANE_SLOTS.len());
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        assert!(UnitDirectReduceMicrokernel::best(&hw, &item, &settings[0]).is_ok());
    }
}
