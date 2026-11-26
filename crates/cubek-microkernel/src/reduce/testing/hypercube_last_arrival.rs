use crate::reduce::cube::{CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings};
use crate::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LANE_SLOT_COUNT, LaunchedRows, LineDirection, MicrokernelLauncher, bound,
};
use crate::reduce::{BATCH, REDUCED, ReduceCarrierFamily, ReduceWorkItem, SLOTS};
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Dimension, Levels, Partitioning, Space, Tile, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The parts a hypercube splits the reduced dimension into, a cube a part.
const PARTS: Dimension = Dimension(3);
/// The lines of a row's partials, every part's, which only they run along.
const PARTIAL_LINES: Dimension = Dimension(4);

/// `parts` cubes of `planes` planes, each reducing its part of the row, the last to finish merging
/// the parts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HypercubeLastArrivalMicrokernelLauncher {
    planes: usize,
    parts: usize,
}

impl HypercubeLastArrivalMicrokernelLauncher {
    pub(crate) const fn new(planes: usize, parts: usize) -> Self {
        Self { planes, parts }
    }
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
            LineDirection::Along => ([BATCH, PARTIAL_LINES], [rows.row_count, lines]),
            LineDirection::Across => ([PARTIAL_LINES, BATCH], [lines, rows.row_count]),
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
        rows.plane_lane_count * self.planes
    }

    fn space(&self, rows: &LaunchedRows) -> Space {
        Space::new(&[
            (BATCH, rows.row_count),
            (REDUCED, rows.value_count / self.parts),
            (SLOTS, rows.cells()),
            (PARTS, self.parts),
            (PARTIAL_LINES, self.partial_lines(rows)),
        ])
    }

    fn bound_input(&self, input: TensorArg, rows: &LaunchedRows) -> TileTensorLaunch {
        match rows.direction() {
            LineDirection::Along => rows.bound_input(input, &[BATCH, PARTS, REDUCED]),
            LineDirection::Across => rows.bound_input(input, &[PARTS, REDUCED, BATCH]),
        }
    }

    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        Levels::leaf(&[
            (BATCH, rows.owned()),
            (REDUCED, rows.value_count / self.parts),
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
            bound(
                arg,
                element,
                rows.line_lane_count,
                &dimensions,
                Boundary::Zero,
            )
        };
        let arrivals = TensorHandle::zeros(launch.client, [rows.row_count], counter);
        let part_lines = rows.value_count / self.parts / rows.row_lanes();
        let indexed = rows.item.operation.carrier().carries_indices();
        let part = CubeStagedReduceMicrokernel::new(
            rows.item,
            rows.plane_lane_count,
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
                rows.line_lane_count,
                rows.cells(),
                launch.input,
                rows.bound_out(launch.out),
                zeroed(rows.item.accumulator),
                indexed.then(|| zeroed(counter)).into(),
                arrivals.binding().into_tensor_arg().into_buffer_arg(),
                launch.launcher.partitioning_arg(),
                self.parts,
                part_lines,
                rows.value_count,
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
/// - `out`: the hypercube last arrival's cells, one a row (`k` for a top-k).
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
    let pi: ComptimeOption<Tile> = partial_indices.map(|indices| indices.tile_mut(&partitioning));

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
