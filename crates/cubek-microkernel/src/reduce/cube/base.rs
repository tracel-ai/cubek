use super::microkernel::{cube_staged_reduce_microkernel, cube_staged_reduce_part};
use crate::reduce::{
    REDUCED, ReduceAccumulator, ReduceOperand, ReduceWorkItem, TOP_K_LANE_SLOTS, lines_across,
    units_across,
};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Dimension, Levels, Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A row reduced by one cube: its units read consecutive lines, each accumulating every `units`-th into
/// a partial of its own; each plane's partials meet in its instructions, then the planes' in
/// shared memory, one slot of the carrier a plane. Where its lines run across the reduced
/// dimension, each plane's units lie across [`lines_across`](Self::lines_across) lines of rows
/// instead, a column of them for each, and a slot a column a plane.
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
    /// Lines of rows each plane's units lie across side by side
    /// ([`across_rows`](Self::across_rows)).
    #[cube(comptime)]
    lines_across: usize,
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
            lines_across: 1,
        }
    }

    /// This microkernel with each plane's units across as many lines of rows, `width` wide, as
    /// one memory transaction of `hw` holds, where its lines run across the reduced dimension and
    /// `rows` rows there are; one line of rows where its lines run along it.
    pub fn across_rows(self, hw: &Hardware, width: usize, rows: usize) -> Self {
        Self {
            lines_across: lines_across(hw, &self.item, width, rows.div_ceil(width), self.lanes),
            ..self
        }
    }

    pub fn lines_across(&self) -> usize {
        self.lines_across
    }

    /// The levels below its cube from `owner`, the box of rows one line wide a row's window
    /// holds: the box, read flat by all of the cube's units, or
    /// [`lines_across`](Self::lines_across) of them side by side, each plane's units laid
    /// across them and in turns along the reduced dimension, the planes after one another along
    /// it, every step of which the cube walks.
    pub fn levels(&self, owner: &[(Dimension, usize)]) -> Levels {
        match self.lines_across {
            1 => Levels::leaf(owner),
            columns => units_across(owner, columns, self.lanes / columns)
                .planes(&[(REDUCED, self.planes)])
                .walk_every(&[REDUCED]),
        }
    }

    /// The units its levels make a plane of the launch: the whole cube where they state no
    /// plane, its planes' lanes where they split the units into planes.
    pub fn launch_plane_width(&self) -> usize {
        match self.lines_across {
            1 => self.units(),
            _ => self.lanes,
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

    /// As the unit microkernel's [`widest_line`](crate::reduce::unit::UnitDirectReduceMicrokernel::widest_line), at this one's lane slots.
    ///
    /// # Panics
    ///
    /// As the unit microkernel's.
    pub fn widest_line(&self) -> Option<usize> {
        self.item
            .operation
            .widest_line_at_lane_slots(self.lane_slots)
    }
}

#[cube]
impl CubeStagedReduceMicrokernel {
    /// `cube`'s rows of `input` reduced into `out` and `indices` at the same region
    /// (`cube_staged_reduce_microkernel`), `cube` a region of the levels
    /// [`levels`](CubeStagedReduceMicrokernel::levels) states.
    ///
    /// # Panics
    ///
    /// Where its item writes the indices beside the values and `indices` is none.
    pub fn execute<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
        &self,
        cube: &Region,
        input: &Tile,
        out: &Tile,
        indices: &ComptimeOption<Tile>,
    ) {
        cube_staged_reduce_microkernel::<In, Acc, Out, Idx, N>(
            cube,
            input,
            out,
            indices,
            comptime!(CubeStagedReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes,
                lane_slots: self.lane_slots,
                lines_across: self.lines_across
            }),
        )
    }

    /// The first `count` lines of `input` accumulated by the cube into `C`, the line at `l` read
    /// as line `first_line + l` of its row: a part of a row that cubes split, which the cube's
    /// first unit is left holding (`cube_staged_reduce_part`).
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
                lane_slots: self.lane_slots,
                lines_across: self.lines_across
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
    /// carrier a plane and a column of rows, which its planes meet through.
    pub fn shared_memory_bytes_at_width(&self, width: usize) -> usize {
        let slot = self
            .item
            .operation
            .carrier()
            .bytes(width, self.item.accumulator.size());
        self.planes * self.lines_across * slot
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
            "{:?} {:?} lanes={} planes={} lines_across={}",
            self.item.operation, self.item.lines_along, self.lanes, self.planes, self.lines_across
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
                lines_across: 1,
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
        estimate.work = Some(self.item.unit_work());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;
    use crate::reduce::plane::{
        PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings,
    };
    use crate::reduce::unit::{UnitDirectReduceMicrokernel, UnitDirectReduceMicrokernelSettings};
    use cubek_std::devices;

    fn item() -> ReduceWorkItem {
        ReduceWorkItem::along_f32(ReduceOperation::ArgMax)
    }

    /// The unit microkernel's first setting.
    fn unit_settings() -> UnitDirectReduceMicrokernelSettings {
        UnitDirectReduceMicrokernelSettings {
            lane_slots: TOP_K_LANE_SLOTS[0],
        }
    }

    /// The plane microkernel's first setting.
    fn plane_settings() -> PlaneShuffleReduceMicrokernelSettings {
        PlaneShuffleReduceMicrokernelSettings {
            lane_slots: TOP_K_LANE_SLOTS[0],
        }
    }

    /// The cube microkernel's first setting.
    fn cube_settings() -> CubeStagedReduceMicrokernelSettings {
        CubeStagedReduceMicrokernelSettings {
            planes: CubeStagedReduceMicrokernelSettings::PLANES[0],
            lane_slots: TOP_K_LANE_SLOTS[0],
        }
    }

    /// The unit runs on a device without planes.
    #[test]
    fn the_unit_runs_without_planes() {
        let hw = devices::wgpu_no_subgroups();
        assert!(UnitDirectReduceMicrokernel::best(&hw, &item(), &unit_settings()).is_ok());
    }

    /// The plane declines on a device without planes.
    #[test]
    fn the_plane_declines_without_planes() {
        let hw = devices::wgpu_no_subgroups();
        assert!(PlaneShuffleReduceMicrokernel::best(&hw, &item(), &plane_settings()).is_err());
    }

    /// The plane declines on a device whose plane width is a range it does not commit to.
    #[test]
    fn the_plane_declines_on_a_range_of_plane_widths() {
        let hw = devices::rdna_range();
        assert!(PlaneShuffleReduceMicrokernel::best(&hw, &item(), &plane_settings()).is_err());
    }

    /// The plane runs on a device that commits to a plane width.
    #[test]
    fn the_plane_runs_on_a_committed_plane_width() {
        let hw = devices::h100();
        assert!(PlaneShuffleReduceMicrokernel::best(&hw, &item(), &plane_settings()).is_ok());
    }

    /// The cube declines on a device whose plane width is a range it does not commit to.
    #[test]
    fn the_cube_declines_on_a_range_of_plane_widths() {
        let hw = devices::rdna_range();
        assert!(CubeStagedReduceMicrokernel::best(&hw, &item(), &cube_settings()).is_err());
    }

    /// The cube's lanes are the width its device commits to.
    #[test]
    fn the_cubes_lanes_are_the_committed_plane_width() {
        let hw = devices::m2_pro();
        let cube = CubeStagedReduceMicrokernel::best(&hw, &item(), &cube_settings())
            .expect("an M2 Pro commits to a plane width");
        assert_eq!(Some(cube.lanes as u32), hw.plane_dim().ok());
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

    /// A cube's units are its planes' lanes.
    #[test]
    fn a_cubes_units_are_its_planes_lanes() {
        const LANES: usize = 32;
        const PLANES: usize = 4;
        const LANE_SLOTS: usize = 32;
        let item = ReduceWorkItem::along_f32(ReduceOperation::ArgMax);
        let cube = CubeStagedReduceMicrokernel::new(item, LANES, PLANES, LANE_SLOTS);
        assert_eq!(cube.units(), LANES * PLANES);
    }

    /// A cube's shared memory is a carrier's slot a plane.
    #[test]
    fn a_cube_holds_a_slot_a_plane() {
        const LANES: usize = 32;
        const PLANES: usize = 4;
        const LANE_SLOTS: usize = 32;
        const WIDTH: usize = 4;
        let item = ReduceWorkItem::along_f32(ReduceOperation::ArgMax);
        let cube = CubeStagedReduceMicrokernel::new(item, LANES, PLANES, LANE_SLOTS);
        let slot = item
            .operation
            .carrier()
            .bytes(WIDTH, item.accumulator.size());
        assert_eq!(cube.shared_memory_bytes_at_width(WIDTH), PLANES * slot);
    }

    /// Its settings are each plane count with each lane-slot count.
    #[test]
    fn its_settings_cross_planes_with_lane_slots() {
        let settings = CubeStagedReduceMicrokernel::settings(&devices::m2_pro());
        assert_eq!(
            settings.len(),
            CubeStagedReduceMicrokernelSettings::PLANES.len() * TOP_K_LANE_SLOTS.len()
        );
    }

    /// Its first setting is its first plane count.
    #[test]
    fn its_first_setting_is_its_first_plane_count() {
        let settings = CubeStagedReduceMicrokernel::settings(&devices::m2_pro());
        assert_eq!(
            settings[0].planes,
            CubeStagedReduceMicrokernelSettings::PLANES[0]
        );
    }
}
