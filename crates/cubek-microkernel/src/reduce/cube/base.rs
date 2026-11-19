use super::microkernel::{cube_staged_reduce_microkernel, cube_staged_reduce_part};
use crate::reduce::{ReduceAccumulator, ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOTS};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row reduced by one cube: its units read consecutive lines, each accumulating every `units`-th into
/// a partial of its own; each plane's partials meet in its instructions, then the planes' in
/// shared memory, one slot of the carrier a plane.
///
/// It runs where the device commits to a plane width, as the plane's does.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// Units of a plane, as wide as the device commits to.
    #[cube(comptime)]
    lanes: usize,
    /// Planes of the cube.
    #[cube(comptime)]
    planes: usize,
    /// As [`UnitDirectReduceMicrokernelSettings::lane_slots`](crate::reduce::unit::UnitDirectReduceMicrokernelSettings::lane_slots).
    #[cube(comptime)]
    lane_slots: usize,
}

impl CubeStagedReduceMicrokernel {
    /// A cube of `planes` planes of `lanes` units reducing `item`, keeping `lane_slots` top-k
    /// slots across a line.
    pub fn new(item: ReduceWorkItem, lanes: usize, planes: usize, lane_slots: usize) -> Self {
        Self {
            item,
            lanes,
            planes,
            lane_slots,
        }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn planes(&self) -> usize {
        self.planes
    }

    /// The units of its cube: its planes, each `lanes` wide.
    pub fn units(&self) -> usize {
        self.lanes * self.planes
    }

    /// The widest line its carrier keeps apart at its lane slots, where the carrier bounds it.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slots(self.lane_slots)
    }
}

#[cube]
impl CubeStagedReduceMicrokernel {
    /// One cube's row of `input` reduced into `out` and `indices` (`cube_staged_reduce_microkernel`).
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
        cube_staged_reduce_microkernel::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            comptime!(CubeStagedReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes,
                lane_slots: self.lane_slots
            }),
        )
    }

    /// The first `count` lines of `input` accumulated by the cube into `C`, the line at `l` read
    /// as line `first_line + l` of its row: a part of a row that cubes split, which every unit of
    /// the cube is left holding (`cube_staged_reduce_part`).
    pub fn reduce_part<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
        &self,
        input: &Tile,
        first_line: usize,
        count: usize,
    ) -> C {
        cube_staged_reduce_part::<In, Acc, N, C>(
            input,
            first_line,
            count,
            comptime!(CubeStagedReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes,
                lane_slots: self.lane_slots
            }),
        )
    }
}

/// How many planes a cube runs, and its top-k lane slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedReduceMicrokernelSettings {
    pub planes: usize,
    /// As [`UnitDirectReduceMicrokernelSettings::lane_slots`](crate::reduce::unit::UnitDirectReduceMicrokernelSettings::lane_slots).
    pub lane_slots: usize,
}

impl CubeStagedReduceMicrokernelSettings {
    /// The planes worth trying: a cube of four, then of eight.
    pub const PLANES: [usize; 2] = [4, 8];
}

impl CubeStagedReduceMicrokernel {
    /// Bytes of shared memory it allocates with its lines `width` lanes wide: one slot of the
    /// carrier a plane, which its planes meet through.
    pub fn shared_memory_bytes_at_width(&self, width: usize) -> usize {
        let slot = self
            .item
            .operation
            .carrier()
            .bytes(width, self.item.accumulator.size());
        self.planes * slot
    }
}

impl core::fmt::Display for CubeStagedReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes={} lane_slots={}", self.planes, self.lane_slots)
    }
}

impl core::fmt::Display for CubeStagedReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} {:?} lanes={} planes={}",
            self.item.operation, self.item.layout, self.lanes, self.planes
        )
    }
}

impl Microkernel for CubeStagedReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeStagedReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube reduce needs a plane width the device commits to, and a cube that holds its planes";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernelSettings::PLANES
            .into_iter()
            .flat_map(|planes| {
                TOP_K_LANE_SLOTS
                    .map(|lane_slots| CubeStagedReduceMicrokernelSettings { planes, lane_slots })
            })
            .collect()
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let planes = settings.planes;
        let lane_slots = settings.lane_slots;
        let fits = planes <= hw.full_cube_planes();
        hw.plane_dim()
            .ok()
            .filter(move |_| fits)
            .into_iter()
            .map(move |lanes| Self {
                item: *item,
                lanes: lanes as usize,
                planes,
                lane_slots,
            })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        vectorization.require(self.item.requirement());
    }

    /// Each unit holds its carrier and one line; the cube a slot of the carrier a plane.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::plane::{
        PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings,
    };
    use crate::reduce::unit::{UnitDirectReduceMicrokernel, UnitDirectReduceMicrokernelSettings};
    use crate::reduce::{ReduceLayout, ReduceOperation};
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    fn item() -> ReduceWorkItem {
        ReduceWorkItem {
            operation: ReduceOperation::ArgMax,
            input: F32,
            accumulator: F32,
            layout: ReduceLayout::Along,
            with_indices: false,
        }
    }

    /// The unit runs everywhere; the plane and the cube only where the device commits to a plane
    /// width, and decline with their reason where it does not.
    #[test]
    fn the_plane_and_the_cube_need_a_width_the_device_commits_to() {
        for hw in [
            devices::strix_halo(),
            devices::m2_pro(),
            devices::h100(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
        ] {
            let commits = hw.plane_dim().is_ok();
            assert!(
                UnitDirectReduceMicrokernel::best(
                    &hw,
                    &item(),
                    &UnitDirectReduceMicrokernelSettings {
                        lane_slots: TOP_K_LANE_SLOTS[0],
                    }
                )
                .is_ok()
            );
            assert_eq!(
                PlaneShuffleReduceMicrokernel::best(
                    &hw,
                    &item(),
                    &PlaneShuffleReduceMicrokernelSettings {
                        lane_slots: TOP_K_LANE_SLOTS[0],
                    }
                )
                .is_ok(),
                commits,
                "{hw:?}"
            );
            for settings in CubeStagedReduceMicrokernel::settings(&hw) {
                let cube = CubeStagedReduceMicrokernel::best(&hw, &item(), &settings);
                assert_eq!(cube.is_ok(), commits, "{settings} on {hw:?}");
                if let Ok(cube) = cube {
                    assert_eq!(cube.lanes as u32, hw.plane_dim().unwrap());
                }
            }
        }
    }

    /// A cube of more planes than the device holds declines.
    #[test]
    fn a_cube_the_device_cannot_hold_declines() {
        let hw = devices::m2_pro();
        let planes = hw.full_cube_planes() + 1;
        let settings = CubeStagedReduceMicrokernelSettings {
            planes,
            lane_slots: TOP_K_LANE_SLOTS[0],
        };
        assert!(CubeStagedReduceMicrokernel::best(&hw, &item(), &settings).is_err());
    }
}
