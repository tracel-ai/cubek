use super::microkernel::plane_shuffle_reduce_microkernel;
use crate::reduce::{
    ReduceAccumulator, ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOTS, lines_across, plane_levels,
};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Dimension, Levels, Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A row reduced by one plane: its lanes read consecutive lines, each accumulating every `lanes`-th
/// into a partial of its own, and the partials meet across the plane in its instructions. Where
/// its lines run across the reduced dimension, its units lie across
/// [`lines_across`](Self::lines_across) lines of rows instead, a column of them for each, and
/// each column's partials meet within it.
///
/// It runs where the device commits to a plane width: the partials meet across the plane, so a
/// plane launched wider than the device runs would leave part of the row out.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// Units of the plane, as wide as the device commits to.
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    lane_slots: usize,
    /// Lines of rows its units lie across side by side ([`across_rows`](Self::across_rows)).
    #[cube(comptime)]
    lines_across: usize,
}

impl PlaneShuffleReduceMicrokernel {
    /// A plane of `lanes` units reducing `item`, keeping `lane_slots` top-k slots across a line.
    pub fn new(item: ReduceWorkItem, lanes: usize, lane_slots: usize) -> Self {
        Self {
            item,
            lanes,
            lane_slots,
            lines_across: 1,
        }
    }

    /// This microkernel with its units across as many lines of rows, `width` wide, as one memory
    /// transaction of `hw` holds, where its lines run across the reduced dimension and `rows`
    /// rows there are; one line of rows where its lines run along it.
    ///
    /// # Panics
    ///
    /// Where `width` is zero.
    pub fn across_rows(self, hw: &Hardware, width: usize, rows: usize) -> Self {
        Self {
            lines_across: lines_across(hw, &self.item, width, rows.div_ceil(width), self.lanes),
            ..self
        }
    }

    pub fn lines_across(&self) -> usize {
        self.lines_across
    }

    /// The levels below its plane from `owner`, the box of rows one line wide a row's window
    /// holds: the box, read flat, or [`lines_across`](Self::lines_across) of them side by side,
    /// the plane's units laid across them and in turns along the reduced dimension, every step
    /// of which the plane walks.
    pub fn levels(&self, owner: &[(Dimension, usize)]) -> Levels {
        plane_levels(owner, self.lines_across, self.lanes)
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// As [`UnitDirectReduceMicrokernel::widest_line`](crate::reduce::unit::UnitDirectReduceMicrokernel::widest_line).
    ///
    /// # Panics
    ///
    /// As it does.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slots(self.lane_slots)
    }
}

#[cube]
impl PlaneShuffleReduceMicrokernel {
    /// `plane`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`plane_shuffle_reduce_microkernel`), `plane` a region of the levels
    /// [`levels`](PlaneShuffleReduceMicrokernel::levels) states.
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
        C: ReduceAccumulator<Acc, N> + Clone,
    >(
        &self,
        plane: &Region,
        input: &Tile,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        plane_shuffle_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneShuffleReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                lane_slots: self.lane_slots,
                lines_across: self.lines_across
            }),
        )
    }
}

/// What a measurement settles for a plane's microkernel: its top-k lane slots. Its lanes are the
/// device's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleReduceMicrokernelSettings {
    pub lane_slots: usize,
}

impl core::fmt::Display for PlaneShuffleReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lane_slots={}", self.lane_slots)
    }
}

impl core::fmt::Display for PlaneShuffleReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} {:?} lanes={} lines_across={}",
            self.item.operation, self.item.lines_along, self.lanes, self.lines_across
        )
    }
}

impl Microkernel for PlaneShuffleReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = PlaneShuffleReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the plane reduce needs a plane width the device commits to";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        TOP_K_LANE_SLOTS
            .map(|lane_slots| PlaneShuffleReduceMicrokernelSettings { lane_slots })
            .to_vec()
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        hw.plane_dim().ok().into_iter().map(|lanes| Self {
            item: *item,
            lanes: lanes as usize,
            lane_slots: settings.lane_slots,
            lines_across: 1,
        })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        vectorization.require(self.item.requirement());
    }

    /// Each lane holds its carrier and one line, at least.
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
    use crate::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Whether the plane microkernel runs on `hw`, at its first setting.
    fn runs_on(hw: &Hardware) -> bool {
        let item = ReduceWorkItem::along_f32(ReduceOperation::ArgMax);
        let settings = PlaneShuffleReduceMicrokernel::settings(hw)[0];
        PlaneShuffleReduceMicrokernel::best(hw, &item, &settings).is_ok()
    }

    #[test]
    fn it_declines_without_planes() {
        assert!(!runs_on(&devices::wgpu_no_subgroups()));
    }

    /// A device whose plane width is a range commits to none of it.
    #[test]
    fn it_declines_on_a_range_of_plane_widths() {
        assert!(!runs_on(&devices::rdna_range()));
    }

    #[test]
    fn it_runs_on_a_committed_plane_width() {
        assert!(runs_on(&devices::h100()));
    }

    /// Where its lines run across the rows, its units lie across a transaction of them.
    #[test]
    fn lines_across_the_rows_lie_a_transaction_across() {
        const LANES: usize = 32;
        const LANE_SLOTS: usize = 32;
        const WIDTH: usize = 4;
        const ROWS: usize = 64;
        let hw = devices::m2_pro();
        let item = ReduceWorkItem {
            lines_along: crate::reduce::BATCH,
            ..ReduceWorkItem::along_f32(ReduceOperation::Sum)
        };
        let plane = PlaneShuffleReduceMicrokernel::new(item, LANES, LANE_SLOTS);
        assert_eq!(
            plane.across_rows(&hw, WIDTH, ROWS).lines_across(),
            hw.transaction_bytes() / (WIDTH * size_of::<f32>())
        );
    }

    /// A candidate's lanes are the plane width the device commits to.
    #[test]
    fn its_lanes_are_the_devices_plane_width() {
        let hw = devices::m2_pro();
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let settings = PlaneShuffleReduceMicrokernel::settings(&hw)[0];
        let plane = PlaneShuffleReduceMicrokernel::best(&hw, &item, &settings)
            .expect("an M2 Pro commits to a plane width");
        assert_eq!(
            plane.lanes() as u32,
            hw.plane_dim().expect("a committed width")
        );
    }

    /// Where its lines run along the row, its units read one line of rows.
    #[test]
    fn lines_along_the_row_lie_one_across() {
        const LANES: usize = 32;
        const LANE_SLOTS: usize = 32;
        const WIDTH: usize = 4;
        const ROWS: usize = 64;
        let plane = PlaneShuffleReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANES,
            LANE_SLOTS,
        );
        assert_eq!(
            plane
                .across_rows(&devices::m2_pro(), WIDTH, ROWS)
                .lines_across(),
            1
        );
    }

    /// Only a top-k bounds the line.
    #[test]
    fn its_widest_line_is_its_carriers() {
        const LANES: usize = 32;
        const LANE_SLOTS: usize = 32;
        let sum = PlaneShuffleReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANES,
            LANE_SLOTS,
        );
        assert_eq!(sum.widest_line(), None);
    }

    fn sum() -> PlaneShuffleReduceMicrokernel {
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let lanes = 32;
        PlaneShuffleReduceMicrokernel::new(item, lanes, TOP_K_LANE_SLOTS[0])
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
        assert_eq!(
            sum().to_string(),
            "Sum Dimension(1) lanes=32 lines_across=1"
        );
    }
}
