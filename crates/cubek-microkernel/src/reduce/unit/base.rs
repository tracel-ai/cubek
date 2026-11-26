use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{
    Accumulable, REDUCED, ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOT_COUNTS, accumulate_read,
};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A row reduced by one unit alone: it reads every line of the row and accumulates each as it lands.
/// Across rows, its lines hold a row a lane, so one unit reduces `width` rows at once.
///
/// It asks nothing of the plane, so it runs on every device.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lane_slots: usize,
}

impl UnitDirectReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, lane_slots: usize) -> Self {
        Self { item, lane_slots }
    }

    /// The widest line its carrier keeps apart at its lane slots, where the carrier bounds it.
    ///
    /// # Panics
    ///
    /// Where its item breaks an Assumption of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem): a top-k that keeps no value.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slot_count(self.lane_slots)
    }
}

#[cube]
impl UnitDirectReduceMicrokernel {
    /// One unit's row of `input` reduced into `out` and `indices` (`unit_direct_reduce_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - `indices` holds a tile exactly where the item writes the indices beside the values
    ///   ([`ReduceWrites::Both`](crate::reduce::ReduceWrites::Both)).
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where an Assumption, or one of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s, is broken.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: Accumulable<Acc, N>,
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
    /// ([`ReduceOperation::widest_line_at_lane_slot_count`](crate::reduce::ReduceOperation::widest_line_at_lane_slot_count)).
    pub lane_slot_count: usize,
}

impl core::fmt::Display for UnitDirectReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lane_slots={}", self.lane_slot_count)
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
        TOP_K_LANE_SLOT_COUNTS
            .map(|lane_slot_count| UnitDirectReduceMicrokernelSettings { lane_slot_count })
            .to_vec()
    }

    fn candidates(
        _hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self {
            item: *item,
            lane_slots: settings.lane_slot_count,
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

/// One unit's row: every line of `input`, the unit's window of the rows it reduces, read in
/// `In`, mapped and accumulated in `Acc` into its carrier, and the total written to `out`, its window of the
/// output, in `Out`.
#[cube]
pub(crate) fn unit_direct_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
) {
    let monoid = microkernel.item.operation.monoid();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));

    let lines = input.flat::<In, N>();
    let count = lines.shape();

    let first_line = 0usize;
    let mut acc = C::identity(monoid);
    for line in 0..count {
        accumulate_read::<In, Acc, N, C>(
            &mut acc,
            lines.read(line),
            first_line,
            line,
            row_lanes,
            microkernel.item,
        );
    }

    unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
        acc,
        out,
        indices,
        count * row_lanes,
        comptime!(UnitCarrierWriteMicrokernel::new(
            microkernel.item,
            row_lanes
        )),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::testing::UnitDirectMicrokernelLauncher;
    use crate::reduce::{ReduceOperation, TOP_K_LANE_SLOT_COUNTS};
    use cubek_std::devices;

    #[test]
    fn a_sum_bounds_no_line() {
        const LANE_SLOT_COUNT: usize = 32;
        let sum = UnitDirectReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_SLOT_COUNT,
        );
        assert_eq!(sum.widest_line(), None);
    }

    #[test]
    fn a_top_k_bounds_the_line_to_its_slots_over_k() {
        const LANE_SLOT_COUNT: usize = 32;
        const KEPT_COUNT: usize = 4;
        let top_k = UnitDirectReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::TopK { k: KEPT_COUNT }),
            LANE_SLOT_COUNT,
        );
        assert_eq!(top_k.widest_line(), Some(LANE_SLOT_COUNT / KEPT_COUNT));
    }

    /// One setting a lane-slot count.
    #[test]
    fn it_lists_a_setting_a_lane_slot_count() {
        let hw = devices::m2_pro();
        assert_eq!(
            UnitDirectReduceMicrokernel::settings(&hw).len(),
            TOP_K_LANE_SLOT_COUNTS.len()
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
        UnitDirectReduceMicrokernel::new(item, TOP_K_LANE_SLOT_COUNTS[0])
    }

    #[test]
    fn along_the_row_the_output_is_written_a_cell_at_a_time() {
        let mut vectorization = Vectorization::new();
        sum().vectorize(&(), &mut vectorization);
        assert_eq!(vectorization.width(ReduceOperand::Out, |_| 8), 1);
    }

    estimates_its_items_work!(sum());

    #[test]
    fn it_displays_what_it_compiles() {
        assert_eq!(sum().to_string(), "Sum Dimension(1)");
    }

    /// The unit's rows 37 values long along the reduced dimension, a value a line.
    mod along_37_long_one_wide {
        use super::*;
        every_operation!(UnitDirectMicrokernelLauncher, Along, 37, 1);
    }

    /// Rows 64 values long along the reduced dimension, four values a line.
    mod along_64_long_four_wide {
        use super::*;
        every_operation!(UnitDirectMicrokernelLauncher, Along, 64, 4);
    }

    /// The unit's rows 37 values long across the reduced dimension, a row a line.
    mod across_37_long_one_wide {
        use super::*;
        every_operation!(UnitDirectMicrokernelLauncher, Across, 37, 1);
    }

    /// The unit's rows 300 values long across the reduced dimension, each lane its own row.
    mod across_300_long_four_wide {
        use super::*;
        every_operation!(UnitDirectMicrokernelLauncher, Across, 300, 4);
    }

    /// The unit's edge rows, 64 values long along the reduced dimension, four values a line.
    mod edges_along {
        use super::*;
        edge_rows!(UnitDirectMicrokernelLauncher, Along, 64, 4);
    }

    /// The unit's edge rows, 300 values long across the reduced dimension, each lane its own row.
    mod edges_across {
        use super::*;
        edge_rows!(UnitDirectMicrokernelLauncher, Across, 300, 4);
    }

    #[test]
    fn its_settings_display_their_lane_slots() {
        let settings = UnitDirectReduceMicrokernelSettings {
            lane_slot_count: 32,
        };
        assert_eq!(settings.to_string(), "lane_slots=32");
    }
}
