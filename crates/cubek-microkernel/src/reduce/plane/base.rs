use super::microkernel::plane_shuffle_reduce_microkernel;
use crate::reduce::{ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOTS};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row reduced by one plane: its lanes read consecutive lines, each accumulating every `lanes`-th
/// into a partial of its own, and the partials meet across the plane in its instructions.
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
    /// As [`UnitDirectReduceMicrokernelSettings::lane_slots`](crate::reduce::unit::UnitDirectReduceMicrokernelSettings::lane_slots).
    #[cube(comptime)]
    lane_slots: usize,
}

impl PlaneShuffleReduceMicrokernel {
    /// A plane of `lanes` units reducing `item`, keeping `lane_slots` top-k slots across a line.
    pub fn new(item: ReduceWorkItem, lanes: usize, lane_slots: usize) -> Self {
        Self {
            item,
            lanes,
            lane_slots,
        }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// The widest line its carrier keeps apart at its lane slots, where the carrier bounds it.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slots(self.lane_slots)
    }
}

#[cube]
impl PlaneShuffleReduceMicrokernel {
    /// One plane's row of `input` reduced into `out` and `indices` (`plane_shuffle_reduce_microkernel`).
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
        plane_shuffle_reduce_microkernel::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            comptime!(PlaneShuffleReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                lane_slots: self.lane_slots
            }),
        )
    }
}

/// What a measurement settles for a plane's microkernel: its top-k lane slots. Its lanes are the
/// device's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleReduceMicrokernelSettings {
    /// As [`UnitDirectReduceMicrokernelSettings::lane_slots`](crate::reduce::unit::UnitDirectReduceMicrokernelSettings::lane_slots).
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
            "{:?} {:?} lanes={}",
            self.item.operation, self.item.layout, self.lanes
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
