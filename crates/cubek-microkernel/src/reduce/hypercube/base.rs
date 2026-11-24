use crate::reduce::cube::{
    CubePartialReduceMicrokernel, CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
    cube_partial_reduce_microkernel,
};
use crate::reduce::read::{UnitPartialsReduceMicrokernel, unit_partials_reduce_microkernel};
use crate::reduce::{Accumulable, REDUCED, ReduceOperand, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
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
/// zero it held before the write.
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
    pub fn shared_memory_bytes_at_width(&self, width: usize) -> usize {
        self.part.shared_memory_bytes_at_width(width) + ARRIVAL_SLOT_BYTES
    }
}

impl core::fmt::Display for HypercubeLastArrivalReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.part)
    }
}

/// The hypercube of each of the cube's candidates, where device memory stays coherent across cubes in a
/// launch, which is how they meet.
fn candidates_meeting_in_device_memory(
    hw: &Hardware,
    item: &ReduceWorkItem,
    settings: &CubeStagedReduceMicrokernelSettings,
) -> impl Iterator<Item = HypercubeLastArrivalReduceMicrokernel> {
    let meets = hw.properties.features.device_memory_scope;
    CubeStagedReduceMicrokernel::candidates(hw, item, settings)
        .filter(move |_| meets)
        .map(HypercubeLastArrivalReduceMicrokernel::new)
}

impl Microkernel for HypercubeLastArrivalReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeStagedReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the hypercube reduce needs the cube reduce, and one cube's stores reaching another within a launch";

    /// The cube microkernel's.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        candidates_meeting_in_device_memory(hw, item, settings)
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
    use crate::reduce::empty_estimate;
    use crate::reduce::testing::{
        ComponentLaunch, LANE_SLOT_COUNT, LaunchedRows, LineDirection, MicrokernelLauncher, bound,
    };
    use crate::reduce::{BATCH, REDUCED, ReduceCarrierFamily, SLOTS};
    use crate::reduce::{ReduceOperation, ReduceWorkItem};
    use cubecl::ir::{ElemType, UIntKind};
    use cubecl::std::tensor::TensorHandle;
    use cubecl::tile::kind::Boundary;
    use cubecl::tile::launch::Decline;
    use cubecl::tile::{Dimension, Levels, Partitioning, Space, TileTensor};
    use cubek_std::devices;

    /// A hypercube's cube holds its part's shared memory and one arrival slot beside it.
    #[test]
    fn a_cube_holds_its_parts_memory_and_an_arrival_slot() {
        const LANES: usize = 32;
        const PLANES: usize = 4;
        const LANE_SLOTS: usize = 32;
        const WIDTH: usize = 4;
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, LANES, PLANES, LANE_SLOTS);
        let hypercube = HypercubeLastArrivalReduceMicrokernel::new(part);
        assert_eq!(
            hypercube.shared_memory_bytes_at_width(WIDTH),
            part.shared_memory_bytes_at_width(WIDTH) + size_of::<u32>()
        );
    }

    #[test]
    fn its_widest_line_is_its_parts() {
        const LANES: usize = 32;
        const PLANES: usize = 4;
        const LANE_SLOTS: usize = 32;
        let item = ReduceWorkItem::along_f32(ReduceOperation::TopK { k: 4 });
        let part = CubeStagedReduceMicrokernel::new(item, LANES, PLANES, LANE_SLOTS);
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
        assert!(HypercubeLastArrivalReduceMicrokernel::best(&hw, &item, &settings).is_err());
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

    /// The parts a hypercube splits the reduced dimension into, a cube a part.
    const PARTS: Dimension = Dimension(3);
    /// The lines of a row's partials, every part's, which only they run along.
    const PARTIAL_LINES: Dimension = Dimension(4);

    /// `parts` cubes of `planes` planes, each reducing its part of the row, the last to finish merging
    /// the parts.
    #[derive(Debug, Clone, Copy)]
    struct HypercubeLastArrivalMicrokernelLauncher {
        planes: usize,
        parts: usize,
    }

    impl HypercubeLastArrivalMicrokernelLauncher {
        /// Lines of a row's partials, each `width` lanes of one row along the reduced dimension.
        fn partial_lines(&self, rows: &LaunchedRows) -> usize {
            self.parts * rows.cells() * rows.row_lanes()
        }

        /// The partials' dimensions, outermost first, and their shape.
        fn partials(&self, rows: &LaunchedRows) -> ([Dimension; 2], [usize; 2]) {
            let lines = self.partial_lines(rows);
            match rows.direction() {
                LineDirection::Along => ([BATCH, PARTIAL_LINES], [rows.rows, lines]),
                LineDirection::Across => ([PARTIAL_LINES, BATCH], [lines, rows.rows]),
            }
        }
    }

    impl MicrokernelLauncher for HypercubeLastArrivalMicrokernelLauncher {
        fn parts(&self) -> usize {
            self.parts
        }

        fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
            let settings = CubeStagedReduceMicrokernelSettings {
                plane_count: self.planes,
                lane_slot_count: LANE_SLOT_COUNT,
            };
            HypercubeLastArrivalReduceMicrokernel::best(hw, item, &settings)
                .map(|hypercube| hypercube.part().lanes())
        }

        fn launch_plane_width(&self, _hw: &Hardware, rows: &LaunchedRows) -> usize {
            rows.lanes * self.planes
        }

        fn space(&self, rows: &LaunchedRows) -> Space {
            Space::new(&[
                (BATCH, rows.rows),
                (REDUCED, rows.reduced / self.parts),
                (SLOTS, rows.cells()),
                (PARTS, self.parts),
                (PARTIAL_LINES, self.partial_lines(rows)),
            ])
        }

        fn input_dimensions(&self, rows: &LaunchedRows) -> Vec<Dimension> {
            match rows.direction() {
                LineDirection::Along => vec![BATCH, PARTS, REDUCED],
                LineDirection::Across => vec![PARTS, REDUCED, BATCH],
            }
        }

        fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
            Levels::leaf(&[
                (BATCH, rows.owned()),
                (REDUCED, rows.reduced / self.parts),
                (SLOTS, rows.cells()),
                (PARTS, 1),
                (PARTIAL_LINES, self.partial_lines(rows)),
            ])
            .cubes(&[BATCH, PARTS])
        }

        fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
            let elements = launch.elements();
            let rows = launch.rows;
            let (dimensions, shape) = self.partials(&rows);
            let counter = ElemType::UInt(UIntKind::U32);
            let zeroed = |element| {
                let zeros = TensorHandle::zeros(launch.client, shape, element);
                let arg = zeros.binding().into_tensor_arg();
                bound(arg, element, rows.width, &dimensions, Boundary::Zero)
            };
            let arrivals = TensorHandle::zeros(launch.client, [rows.rows], counter);
            let part_lines = rows.reduced / self.parts / rows.row_lanes();
            let indexed = rows.item.operation.carrier().carries_indices();
            let part = CubeStagedReduceMicrokernel::new(
                rows.item,
                rows.lanes,
                self.planes,
                LANE_SLOT_COUNT,
            );
            // SAFETY: every operand but the counters is bound through the launcher at the problem's
            // own extents, and the kernel indexes no buffer but through its tiles and the counters,
            // one a row, each owner's at its first row.
            unsafe {
                hypercube_reduce_kernel::launch_unchecked::<F>(
                    launch.client,
                    launch.launcher.cube_count(),
                    launch.launcher.cube_dim(),
                    rows.width,
                    rows.cells(),
                    launch.input,
                    launch.out,
                    zeroed(rows.item.accumulator),
                    indexed.then(|| zeroed(counter)).into(),
                    arrivals.binding().into_tensor_arg().into_buffer_arg(),
                    launch.launcher.partitioning_arg(),
                    self.parts,
                    part_lines,
                    rows.reduced,
                    HypercubeLastArrivalReduceMicrokernel::new(part),
                    elements,
                )
            }
        }
    }

    /// Every row's part, reduced by its cube; the row's last cube merges the parts.
    ///
    /// Its operands' leaves:
    /// - `input`: the rows, each split into `parts` equal parts.
    /// - `out`: one tensor of a cell a row (`k` for a top-k).
    /// - `partials`: every part's carrier of every row, in the accumulator's element.
    /// - `partial_indices`: the index of every value `partials` keeps, in `u32`, where it keeps any.
    /// - `arrivals`: one `u32` counter a row, at zero.
    #[cube(launch_unchecked)]
    fn hypercube_reduce_kernel<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        N: Size,
        K: Size,
        F: ReduceCarrierFamily,
    >(
        input: &TileTensor<'_>,
        out: &mut TileTensor<'_>,
        partials: &mut TileTensor<'_>,
        partial_indices: ComptimeOption<&mut TileTensor<'static>>,
        arrivals: &[Atomic<u32>],
        partitioning: Partitioning,
        #[comptime] parts: usize,
        #[comptime] part_lines: usize,
        #[comptime] values: usize,
        #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
        #[define(In, Acc, Out)] _elements: [ElemType; 3],
    ) {
        let x = input.tile(&partitioning);
        let o = out.tile_mut(&partitioning);
        let p = partials.tile_mut(&partitioning);
        let pi: ComptimeOption<Tile> =
            partial_indices.map(|indices| indices.tile_mut(&partitioning));

        for cube in &partitioning {
            let mut row_out = o.at(&cube);
            let mut slots = p.at(&cube);
            let part = cube.origin(PARTS);

            microkernel
                .runtime()
                .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                    &x.at(&cube),
                    &mut row_out,
                    &ComptimeOption::new_None(),
                    &mut slots,
                    &pi.clone().map(|tile| tile.at(&cube)),
                    arrivals,
                    cube.origin(BATCH),
                    part,
                    parts,
                    part * part_lines,
                    values,
                );
        }
    }

    /// Rows 64 values long along the reduced dimension, a value a line, in two parts each shorter than
    /// a cube's units.
    mod along_two_short_parts {
        use super::*;

        const MICROKERNEL: HypercubeLastArrivalMicrokernelLauncher =
            HypercubeLastArrivalMicrokernelLauncher {
                planes: 4,
                parts: 2,
            };

        every_operation!(MICROKERNEL, Along, 64, 1);
    }

    /// Rows 8192 values long along the reduced dimension, four values a line, in eight parts each
    /// several times a cube's units.
    mod along_eight_long_parts {
        use super::*;

        const MICROKERNEL: HypercubeLastArrivalMicrokernelLauncher =
            HypercubeLastArrivalMicrokernelLauncher {
                planes: 4,
                parts: 8,
            };

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
            HypercubeLastArrivalMicrokernelLauncher {
                planes: 4,
                parts: 3,
            };

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
}
