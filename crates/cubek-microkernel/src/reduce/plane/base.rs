use crate::reduce::plane::{
    PlaneColumnsReduceMicrokernel, PlaneLineReduceMicrokernel, plane_columns_reduce_microkernel,
    plane_line_reduce_microkernel,
};
use crate::reduce::{
    Accumulable, ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOT_COUNTS, lines_across, plane_levels,
};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Dimension, Levels, Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A row reduced by one plane: its lanes read the lines in turns, each into a partial of its own, and
/// the partials meet across the plane in its instructions. Where
/// its lines run across the reduced dimension, its units lie across
/// [`lines_across`](Self::lines_across) lines of rows instead, a column of them for each, and
/// each column's partials meet within it.
///
/// Its partials meet across the plane, so it runs where the device commits to a plane width: a plane
/// wider than the one that runs leaves some of them out.
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
    /// # Assumptions
    ///
    /// - `width` is at least one.
    ///
    /// # Panics
    ///
    /// Where its Assumption is broken.
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

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// As [`UnitDirectReduceMicrokernel::widest_line`](crate::reduce::unit::UnitDirectReduceMicrokernel::widest_line).
    ///
    /// # Panics
    ///
    /// Where its item breaks an Assumption of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem): a
    /// top-k that keeps no value.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slot_count(self.lane_slots)
    }
}

#[cube]
impl PlaneShuffleReduceMicrokernel {
    /// `plane`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`plane_shuffle_reduce_microkernel`), `plane` a region of the levels
    /// [`levels`](PlaneShuffleReduceMicrokernel::levels) states.
    ///
    /// # Assumptions
    ///
    /// - Those of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem), of its item.
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where the item breaks
    /// [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s Assumptions on `indices`: a tile handed
    /// where the item writes no indices, none where it writes them, or indices written that its
    /// carrier does not keep.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        Idx: Numeric,
        N: Size,
        C: Accumulable<Acc, N> + Clone,
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
    pub lane_slot_count: usize,
}

impl core::fmt::Display for PlaneShuffleReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lane_slots={}", self.lane_slot_count)
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

/// The plane microkernel at `settings`, where `hw` commits to a plane width.
fn candidate_of(
    hw: &Hardware,
    item: ReduceWorkItem,
    settings: &PlaneShuffleReduceMicrokernelSettings,
) -> Option<PlaneShuffleReduceMicrokernel> {
    let lanes = hw.plane_dim().ok()?;
    Some(PlaneShuffleReduceMicrokernel::new(
        item,
        lanes as usize,
        settings.lane_slot_count,
    ))
}

impl Microkernel for PlaneShuffleReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = PlaneShuffleReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the plane reduce needs a plane width the device commits to";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        TOP_K_LANE_SLOT_COUNTS
            .map(|lane_slot_count| PlaneShuffleReduceMicrokernelSettings { lane_slot_count })
            .to_vec()
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        candidate_of(hw, *item, settings).into_iter()
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

/// One plane's rows, `plane`'s region of `input`, reduced into `out` and `indices` at the same
/// region: read one line of rows wide ([`PlaneLineReduceMicrokernel`]), or
/// [`lines_across`](PlaneShuffleReduceMicrokernel::lines_across) lines of rows wide
/// ([`PlaneColumnsReduceMicrokernel`]).
#[cube]
pub(crate) fn plane_shuffle_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: Accumulable<Acc, N> + Clone,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    match microkernel.lines_across == 1 {
        true => plane_line_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneLineReduceMicrokernel::new(
                microkernel.item,
                microkernel.lanes
            )),
        ),
        false => plane_columns_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            comptime!(PlaneColumnsReduceMicrokernel::new(
                microkernel.item,
                microkernel.lanes,
                microkernel.lines_across
            )),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::plane::PlaneShuffleReduceMicrokernel;
    use crate::reduce::testing::PlaneShuffleMicrokernelLauncher;
    use crate::reduce::{BATCH, ReduceOperation};
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
        const LANE_COUNT: usize = 32;
        const LANE_SLOT_COUNT: usize = 32;
        const LINE_LANE_COUNT: usize = 4;
        const ROW_COUNT: usize = 64;
        let hw = devices::m2_pro();
        let item = ReduceWorkItem {
            lines_along: BATCH,
            ..ReduceWorkItem::along_f32(ReduceOperation::Sum)
        };
        let plane = PlaneShuffleReduceMicrokernel::new(item, LANE_COUNT, LANE_SLOT_COUNT);
        assert_eq!(
            plane
                .across_rows(&hw, LINE_LANE_COUNT, ROW_COUNT)
                .lines_across(),
            hw.transaction_bytes() / (LINE_LANE_COUNT * size_of::<f32>())
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
        const LANE_COUNT: usize = 32;
        const LANE_SLOT_COUNT: usize = 32;
        const LINE_LANE_COUNT: usize = 4;
        const ROW_COUNT: usize = 64;
        let plane = PlaneShuffleReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_COUNT,
            LANE_SLOT_COUNT,
        );
        assert_eq!(
            plane
                .across_rows(&devices::m2_pro(), LINE_LANE_COUNT, ROW_COUNT)
                .lines_across(),
            1
        );
    }

    /// Only a top-k bounds the line.
    #[test]
    fn its_widest_line_is_its_carriers() {
        const LANE_COUNT: usize = 32;
        const LANE_SLOT_COUNT: usize = 32;
        let sum = PlaneShuffleReduceMicrokernel::new(
            ReduceWorkItem::along_f32(ReduceOperation::Sum),
            LANE_COUNT,
            LANE_SLOT_COUNT,
        );
        assert_eq!(sum.widest_line(), None);
    }

    fn sum() -> PlaneShuffleReduceMicrokernel {
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let lanes = 32;
        PlaneShuffleReduceMicrokernel::new(item, lanes, TOP_K_LANE_SLOT_COUNTS[0])
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
        assert_eq!(
            sum().to_string(),
            "Sum Dimension(1) lanes=32 lines_across=1"
        );
    }

    /// The plane's rows 37 values long along the reduced dimension, a value a line.
    mod along_37_long_one_wide {
        use super::*;
        every_operation!(PlaneShuffleMicrokernelLauncher, Along, 37, 1);
    }

    /// Rows 20 values long along the reduced dimension, four values a line: fewer lines than lanes.
    mod along_20_long_four_wide {
        use super::*;
        every_operation!(PlaneShuffleMicrokernelLauncher, Along, 20, 4);
    }

    /// Rows 1024 values long along the reduced dimension, four values a line: many lines a lane.
    mod along_1024_long_four_wide {
        use super::*;
        every_operation!(PlaneShuffleMicrokernelLauncher, Along, 1024, 4);
    }

    /// The plane's rows 37 values long across the reduced dimension, a row a line.
    mod across_37_long_one_wide {
        use super::*;
        every_operation!(PlaneShuffleMicrokernelLauncher, Across, 37, 1);
    }

    /// The plane's rows 300 values long across the reduced dimension, each lane its own row.
    mod across_300_long_four_wide {
        use super::*;
        every_operation!(PlaneShuffleMicrokernelLauncher, Across, 300, 4);
    }

    /// The plane's edge rows, 1024 values long along the reduced dimension, four values a line.
    mod edges_along {
        use super::*;
        edge_rows!(PlaneShuffleMicrokernelLauncher, Along, 1024, 4);
    }

    /// The plane's edge rows, 300 values long across the reduced dimension, each lane its own row.
    mod edges_across {
        use super::*;
        edge_rows!(PlaneShuffleMicrokernelLauncher, Across, 300, 4);
    }

    #[test]
    fn its_settings_display_their_lane_slots() {
        let settings = PlaneShuffleReduceMicrokernelSettings {
            lane_slot_count: 32,
        };
        assert_eq!(settings.to_string(), "lane_slots=32");
    }
}
