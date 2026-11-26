use crate::reduce::cube::{
    CubePartialReduceMicrokernel, CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
    cube_partial_reduce_microkernel,
};
use crate::reduce::read::{UnitPartialsReduceMicrokernel, unit_partials_reduce_microkernel};
use crate::reduce::{Accumulable, REDUCED, ReduceOperand, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::tile::launch::{Decline, Vectorization};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Bytes of the slot of shared memory a cube's first unit hands it its arrival through.
const ARRIVAL_SLOT_BYTES: usize = size_of::<u32>();

/// A row split across cubes, each reducing its part with the cube's microkernel, then storing its
/// partial apart: the last of the row's cubes to finish merges the partials and writes the row.
///
/// It runs where the cube's microkernel runs and where one cube's stores reach another within a
/// launch (`device_memory_scope`), which is how the cubes meet: on WGSL, whose storage barrier
/// orders a workgroup's own memory only, the last cube was seen reading another's partial as the
/// zero it stored before the write.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalReduceMicrokernel {
    /// The cube microkernel each cube reduces its part with.
    #[cube(comptime)]
    part: CubeStagedReduceMicrokernel,
}

impl HypercubeLastArrivalReduceMicrokernel {
    pub fn new(part: CubeStagedReduceMicrokernel) -> Self {
        Self { part }
    }

    /// The widest line its part's carrier keeps apart.
    ///
    /// # Panics
    ///
    /// Where its part's top-k keeps no value.
    pub fn widest_line(&self) -> Option<usize> {
        self.part.widest_line()
    }

    pub fn part(&self) -> CubeStagedReduceMicrokernel {
        self.part
    }
}

#[cube]
impl HypercubeLastArrivalReduceMicrokernel {
    /// One cube's part of a row of `input`, the row's last cube writing it to `out` and `indices` (`hypercube_last_arrival_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where the hypercube's item and `indices` disagree on whether the indices are written beside the values, or
    /// its carrier keeps none of the indices it writes.
    /// Where its part's carrier and `partial_indices` disagree on whether it keeps indices.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and the cube's place in its row, as the microkernel takes them"
    )]
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
        partials: &mut Tile,
        partial_indices: &ComptimeOption<Tile>,
        arrivals: &[Atomic<u32>],
        counter: usize,
        part: usize,
        parts: usize,
        first_line: usize,
        values: usize,
    ) {
        hypercube_last_arrival_reduce_microkernel::<In, Acc, Out, Idx, N, C>(
            input,
            out,
            indices,
            partials,
            partial_indices,
            arrivals,
            counter,
            part,
            parts,
            first_line,
            values,
            comptime!(HypercubeLastArrivalReduceMicrokernel { part: self.part }),
        )
    }
}

impl HypercubeLastArrivalReduceMicrokernel {
    /// Bytes of shared memory each cube allocates with its lines `width` lanes wide: its part's,
    /// and the slot its first unit hands the cube its arrival through.
    pub fn shared_memory_bytes_at_lane_count(&self, width: usize) -> usize {
        self.part.shared_memory_bytes_at_lane_count(width) + ARRIVAL_SLOT_BYTES
    }
}

impl core::fmt::Display for HypercubeLastArrivalReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.part)
    }
}

/// Whether one cube's stores reach another within a launch on `hw`, which is how the cubes meet.
fn meets_in_device_memory(hw: &Hardware) -> bool {
    hw.properties.features.device_memory_scope
}

impl Microkernel for HypercubeLastArrivalReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeStagedReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the last-arrival hypercube needs one cube's stores reaching another within a launch (`device_memory_scope`)";

    /// The cube microkernel's.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let meets = meets_in_device_memory(hw);
        CubeStagedReduceMicrokernel::candidates(hw, item, settings)
            .filter(move |_| meets)
            .map(Self::new)
    }

    /// The hypercube of the cube's best, the cube's own Decline coming through where it has none.
    fn best(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> Result<Self, Decline> {
        let part = CubeStagedReduceMicrokernel::best(hw, item, settings)?;
        if !meets_in_device_memory(hw) {
            return Err(Decline::Unserved { why: Self::DECLINE });
        }
        Ok(Self::new(part))
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        self.part.vectorize(&(), vectorization);
    }

    /// The merging unit holds the merged carrier and one partial, beside two registers of its own:
    /// a lower bound.
    fn estimate(
        &self,
        _item: &ReduceWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(self.part.item().merge_work());
    }
}

/// One cube's part of a row its `parts` cubes split: the cube reduces `input`, its window of the
/// part, with the cube's microkernel, each line indexed `first_line` lines further, its place in
/// the whole row; a part the row's end cuts short reads the edge monoid's identity past it. Its first unit
/// stores the partial as partial `part` of the row's `partials` and `partial_indices`, its
/// carrier's lines a slot apart, and the cube counts itself in at the row's `counter` of
/// `arrivals`. The row's last cube to count merges every partial and writes the
/// row to `out`, and to `indices` where both halves are written, as the cube's microkernel writes
/// a row, a mean dividing by `values`, the values the whole row holds.
///
/// `arrivals` holds a counter for every row, or box of rows, its caller counts apart, at zero; the
/// row's last cube sets it back to zero once it has merged, so the caller's counters serve its
/// next launch.
/// The cube meets on `sync_storage` before it counts, so its partial lands first, and again before
/// the last cube reads the others'.
#[cube]
pub(crate) fn hypercube_last_arrival_reduce_microkernel<
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
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    arrivals: &[Atomic<u32>],
    counter: usize,
    part: usize,
    parts: usize,
    first_line: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
) {
    let cube = microkernel.part;
    cube_partial_reduce_microkernel::<In, Acc, N, C>(
        input,
        partials,
        partial_indices,
        part,
        first_line,
        comptime!(CubePartialReduceMicrokernel::new(cube.strided())),
    );

    // The partial lands at device scope before the cube counts itself in.
    sync_storage();
    let last = counts_in_last(arrivals, counter, parts);
    sync_storage_where_last(last);

    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let merges = last && UNIT_POS == 0;
    if merges {
        unit_partials_reduce_microkernel::<Out, Idx, Acc, N, C>(
            &*partials,
            partial_indices,
            parts,
            out,
            indices,
            values,
            comptime!(UnitPartialsReduceMicrokernel::new(cube.item(), row_lanes)),
        );
    }
    reset_count_where(arrivals, counter, merges);
}

/// Whether the cube is the last of its row's `parts` to count itself in at `counter` of `arrivals`:
/// its first unit counts it in, and the cube meets so every unit answers the same.
#[cube]
fn counts_in_last(arrivals: &[Atomic<u32>], counter: usize, parts: usize) -> bool {
    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if UNIT_POS == 0 {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }

    sync_cube();
    arrived[0] as usize + 1 == parts
}

/// What every other cube of the row stored acquired before it is read, by the cube whose units all
/// hold `last`.
#[cube]
fn sync_storage_where_last(last: bool) {
    if last {
        sync_storage();
    }
}

/// The counter at `counter` of `arrivals` set back to zero by the unit that `merges` the row, which
/// every other cube of the row counted in before: nothing counts there again in this launch.
#[cube]
fn reset_count_where(arrivals: &[Atomic<u32>], counter: usize, merges: bool) {
    if merges {
        arrivals[counter].store(0u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::cube::CubeStagedReduceMicrokernel;
    use crate::reduce::testing::{HypercubeLastArrivalMicrokernelLauncher, LANE_SLOT_COUNT};
    use crate::reduce::{ReduceOperation, ReduceWorkItem, empty_estimate};
    use cubek_std::devices;

    /// A hypercube's cube holds its part's shared memory and one arrival slot beside it.
    #[test]
    fn a_cube_holds_its_parts_memory_and_an_arrival_slot() {
        const LANE_COUNT: usize = 32;
        const PLANE_COUNT: usize = 4;
        const LANE_SLOT_COUNT: usize = 32;
        const LINE_LANE_COUNT: usize = 4;
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, LANE_COUNT, PLANE_COUNT, LANE_SLOT_COUNT);
        let hypercube = HypercubeLastArrivalReduceMicrokernel::new(part);
        assert_eq!(
            hypercube.shared_memory_bytes_at_lane_count(LINE_LANE_COUNT),
            part.shared_memory_bytes_at_lane_count(LINE_LANE_COUNT) + size_of::<u32>()
        );
    }

    #[test]
    fn its_widest_line_is_its_parts() {
        const LANE_COUNT: usize = 32;
        const PLANE_COUNT: usize = 4;
        const LANE_SLOT_COUNT: usize = 32;
        let item = ReduceWorkItem::along_f32(ReduceOperation::TopK { k: 4 });
        let part = CubeStagedReduceMicrokernel::new(item, LANE_COUNT, PLANE_COUNT, LANE_SLOT_COUNT);
        let hypercube = HypercubeLastArrivalReduceMicrokernel::new(part);
        assert_eq!(hypercube.widest_line(), part.widest_line());
    }

    /// Its cubes meet through device memory, so without device memory scope it has no candidate.
    #[test]
    fn it_needs_device_memory_scope() {
        let mut hw = devices::m2_pro();
        devices::features(&mut hw).device_memory_scope = false;
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let settings = HypercubeLastArrivalReduceMicrokernel::settings(&hw)[0];
        assert_eq!(
            HypercubeLastArrivalReduceMicrokernel::best(&hw, &item, &settings),
            Err(Decline::Unserved {
                why: HypercubeLastArrivalReduceMicrokernel::DECLINE
            })
        );
    }

    /// The merged carrier and one partial, a line each, beside two registers.
    #[test]
    fn its_estimate_counts_the_merged_carrier_and_a_partial() {
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, 32, 4, 32);
        let planes = CubePlanes {
            count: 4,
            width: 32,
        };
        let mut estimate = empty_estimate();
        HypercubeLastArrivalReduceMicrokernel::new(part).estimate(
            &item,
            &(),
            planes,
            &mut estimate,
        );
        assert_eq!(estimate.work.map(|work| work.registers), Some(4));
    }

    /// Rows 64 values long along the reduced dimension, a value a line, in two parts each shorter than
    /// a cube's units.
    mod along_two_short_parts {
        use super::*;

        const MICROKERNEL: HypercubeLastArrivalMicrokernelLauncher =
            HypercubeLastArrivalMicrokernelLauncher::new(4, 2);

        every_operation!(MICROKERNEL, Along, 64, 1);
    }

    /// Rows 8192 values long along the reduced dimension, four values a line, in eight parts each
    /// several times a cube's units.
    mod along_eight_long_parts {
        use super::*;

        const MICROKERNEL: HypercubeLastArrivalMicrokernelLauncher =
            HypercubeLastArrivalMicrokernelLauncher::new(4, 8);

        every_operation!(MICROKERNEL, Along, 8192, 4);

        /// The edge rows, in the same eight parts.
        mod edges {
            use super::*;
            edge_rows!(MICROKERNEL, Along, 8192, 4);
        }
    }

    /// Rows 300 values long across the reduced dimension, each lane its own row, in three parts.
    mod across_three_parts {
        use super::*;

        const MICROKERNEL: HypercubeLastArrivalMicrokernelLauncher =
            HypercubeLastArrivalMicrokernelLauncher::new(4, 3);

        every_operation!(MICROKERNEL, Across, 300, 4);

        /// The edge rows, in the same three parts.
        mod edges {
            use super::*;
            edge_rows!(MICROKERNEL, Across, 300, 4);
        }
    }

    /// What its part's cube asks of the operands.
    #[test]
    fn along_the_row_its_output_takes_a_cell_a_row() {
        let (lanes, planes) = (32, 4);
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, lanes, planes, LANE_SLOT_COUNT);
        let mut vectorization = Vectorization::new();
        HypercubeLastArrivalReduceMicrokernel::new(part).vectorize(&(), &mut vectorization);
        assert_eq!(vectorization.width(ReduceOperand::Out, |_| 8), 1);
    }

    #[test]
    fn it_displays_its_part() {
        let (lanes, planes) = (32, 4);
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, lanes, planes, LANE_SLOT_COUNT);
        let hypercube = HypercubeLastArrivalReduceMicrokernel::new(part);
        assert_eq!(hypercube.to_string(), part.to_string());
    }
}
