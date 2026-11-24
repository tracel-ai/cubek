use super::microkernel::unit_direct_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOTS};
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
    ///
    /// # Panics
    ///
    /// Where its carrier is a top-k keeping no value.
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
    /// Where `indices` is there and its item does not write them beside the values, or the other
    /// way around; or where it writes indices its carrier does not keep.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: ReduceAccumulator<Acc, N>,
    >(
        &self,
        input: &Tile,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        unit_direct_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
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

    /// A lower bound: the registers it keeps live.
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

    #[test]
    fn a_sum_bounds_no_line() {
        const LANE_SLOTS: usize = 32;
        let sum = UnitDirectReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_SLOTS,
        );
        assert_eq!(sum.widest_line(), None);
    }

    #[test]
    fn a_top_k_bounds_the_line_to_its_slots_over_k() {
        const LANE_SLOTS: usize = 32;
        const K: usize = 4;
        let top_k = UnitDirectReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::TopK { k: K }),
            LANE_SLOTS,
        );
        assert_eq!(top_k.widest_line(), Some(LANE_SLOTS / K));
    }

    /// One setting a lane-slot count.
    #[test]
    fn it_lists_a_setting_a_lane_slot_count() {
        let hw = devices::m2_pro();
        assert_eq!(
            UnitDirectReduceMicrokernel::settings(&hw).len(),
            TOP_K_LANE_SLOTS.len()
        );
    }

    #[test]
    fn it_runs_on_a_device_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let settings = UnitDirectReduceMicrokernel::settings(&hw)[0];
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        assert!(UnitDirectReduceMicrokernel::best(&hw, &item, &settings).is_ok());
    }

    fn sum() -> UnitDirectReduceMicrokernel {
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        UnitDirectReduceMicrokernel::new(item, TOP_K_LANE_SLOTS[0])
    }

    #[test]
    fn along_the_row_the_output_is_written_a_cell_at_a_time() {
        let mut vectorization = Vectorization::new();
        sum().vectorize(&(), &mut vectorization);
        assert_eq!(vectorization.width(ReduceOperand::Out, |_| 8), 1);
    }

    #[test]
    fn its_estimate_is_its_items_work() {
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let planes = CubePlanes {
            count: 4,
            width: 32,
        };
        let mut estimate = crate::reduce::empty_estimate();
        sum().estimate(&item, &(), planes, &mut estimate);
        assert_eq!(estimate.work, Some(item.unit_work()));
    }

    #[test]
    fn it_displays_what_it_compiles() {
        assert_eq!(sum().to_string(), "Sum Dimension(1)");
    }
}
