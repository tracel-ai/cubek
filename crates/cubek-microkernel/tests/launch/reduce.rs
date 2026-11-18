//! The reduce microkernels on one launch: `out[ROWS, SLOTS]`, the cells each row writes (one, or
//! `k` for a top-k). Along the reduced dimension, `input[ROWS, REDUCED]`, each row its owner's;
//! across it, `input[REDUCED, ROWS]` and `out[SLOTS, ROWS]`, each owner `width` rows side by side,
//! a row a lane. A hypercube splits the reduced dimension into parts,
//! `input[ROWS, PARTS, REDUCED]` or `input[PARTS, REDUCED, ROWS]`, a cube a part.

use cubecl::ir::UIntKind;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder, UnitCube};
use cubecl::tile::{
    Dimension, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_microkernel::reduce::plane::{
    PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::unit::UnitDirectReduceMicrokernel;
use cubek_microkernel::reduce::{REDUCED, ROWS, ReduceLayout, ReduceWorkItem, TOP_K_LANE_SLOTS};

/// The lane slots the microkernels' settings list.
const LANE_SLOTS: usize = TOP_K_LANE_SLOTS[0];
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The cells a row writes: one, or `k` for a top-k.
const SLOTS: Dimension = Dimension(2);
/// The parts a hypercube splits the reduced dimension into, a cube a part.
const PARTS: Dimension = Dimension(3);
/// The lines of a row's partials, every part's, which only they run along.
const PARTIAL_LINES: Dimension = Dimension(4);

/// Which reduce microkernel a case runs, named for the microkernel, and what its scope holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TestedMicrokernel {
    /// The unit microkernel: every unit reduces its own row.
    UnitDirect,
    /// The plane microkernel: the whole plane.
    PlaneShuffle,
    /// The cube microkernel: a cube of `planes` planes.
    CubeStaged { planes: usize },
    /// The hypercube microkernel: `parts` cubes of `planes` planes, each reducing its part of the
    /// row, the last to finish merging the parts.
    HypercubeLastArrival { planes: usize, parts: usize },
}

/// A reduce microkernel ready to launch on `rows` rows, each `reduced` values long and
/// contiguous, read `width` values a line.
#[derive(Debug, Clone)]
pub struct ReduceLaunch {
    pub item: ReduceWorkItem,
    pub microkernel: TestedMicrokernel,
    pub rows: usize,
    /// Values a row holds.
    pub reduced: usize,
    pub width: usize,
    /// The element the output is written in: the values', or an index's.
    pub out: ElemType,
    plane_width: u32,
}

impl ReduceLaunch {
    /// # Errors
    ///
    /// Why the device runs no such microkernel.
    pub fn new(
        hw: &Hardware,
        item: ReduceWorkItem,
        microkernel: TestedMicrokernel,
        rows: usize,
        reduced: usize,
        width: usize,
        out: ElemType,
    ) -> Result<Self, Refusal> {
        let plane_width = match microkernel {
            TestedMicrokernel::UnitDirect => hw.plane_dim().unwrap_or(1),
            TestedMicrokernel::PlaneShuffle => PlaneShuffleReduceMicrokernel::best(
                hw,
                &item,
                &PlaneShuffleReduceMicrokernelSettings {
                    lane_slots: TOP_K_LANE_SLOTS[0],
                },
            )?
            .lanes() as u32,
            TestedMicrokernel::CubeStaged { planes } => CubeStagedReduceMicrokernel::best(
                hw,
                &item,
                &CubeStagedReduceMicrokernelSettings {
                    planes,
                    lane_slots: LANE_SLOTS,
                },
            )?
            .lanes() as u32,
            TestedMicrokernel::HypercubeLastArrival { planes, .. } => {
                let settings = CubeStagedReduceMicrokernelSettings {
                    planes,
                    lane_slots: LANE_SLOTS,
                };
                HypercubeLastArrivalReduceMicrokernel::best(hw, &item, &settings)?
                    .part()
                    .lanes() as u32
            }
        };
        Ok(Self {
            item,
            microkernel,
            rows,
            reduced,
            width,
            out,
            plane_width,
        })
    }

    /// `out[o, s]` from `input[o, r]` along the reduced dimension, `out[s, o]` from `input[r, o]`
    /// across it, all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(&self, client: &Client, input: TensorArg, out: TensorArg) -> Result<(), Refusal> {
        match self.microkernel {
            TestedMicrokernel::HypercubeLastArrival { planes, parts } => {
                self.launch_split(client, input, out, planes, parts)
            }
            TestedMicrokernel::UnitDirect
            | TestedMicrokernel::PlaneShuffle
            | TestedMicrokernel::CubeStaged { .. } => self.launch_whole(client, input, out),
        }
    }

    /// Every row reduced by one owner.
    fn launch_whole(
        &self,
        client: &Client,
        input: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let cells = self.item.operation.written_per_row();
        let space = Space::new(&[(ROWS, self.rows), (REDUCED, self.reduced), (SLOTS, cells)]);
        // Across, an owner holds a line's worth of rows side by side.
        let (owned, input_dimensions, out_dimensions, out_width) = match self.item.layout {
            ReduceLayout::Along => (1, [ROWS, REDUCED], [ROWS, SLOTS], 1),
            ReduceLayout::Across => (self.width, [REDUCED, ROWS], [SLOTS, ROWS], self.width),
        };
        let leaf = Levels::leaf(&[(ROWS, owned), (REDUCED, self.reduced), (SLOTS, cells)]);
        let (partitioning, cube_dim) = match self.microkernel {
            TestedMicrokernel::UnitDirect => {
                let partitioning = Partitioning::new(
                    space.clone(),
                    leaf.launch_units(
                        &[ROWS],
                        UnitCube {
                            plane_width: self.plane_width as usize,
                            planes: 1,
                        },
                    )
                    .build(),
                );
                let cube_dim = partitioning.cube_dim(self.plane_width);
                (partitioning, cube_dim)
            }
            TestedMicrokernel::PlaneShuffle => {
                let partitioning = Partitioning::new(
                    space.clone(),
                    leaf.launch_planes(&[ROWS], PlaneOrder::new(&[(ROWS, 1)], CubeOrder::RowMajor))
                        .build(),
                );
                let cube_dim = partitioning.cube_dim(self.plane_width);
                (partitioning, cube_dim)
            }
            TestedMicrokernel::CubeStaged { planes } => (
                Partitioning::new(space.clone(), leaf.cubes(&[ROWS]).build()),
                CubeDim::new_2d(self.plane_width, planes as u32),
            ),
            TestedMicrokernel::HypercubeLastArrival { .. } => {
                unreachable!("a hypercube splits its rows apart")
            }
        };
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim,
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        // Past its edge the input reads the identity of the operation's edge monoid; the outputs and
        // the partials are read and written only within theirs.
        let bound = |arg, element, width, dimensions: &[_], boundary| {
            TileTensorLaunch::new(
                arg,
                element,
                width,
                TileSpec::direct(dimensions).boundary(BoundaryPolicy::Every(boundary)),
            )
        };
        let edge = Boundary::Identity(self.item.operation.edge_monoid());
        let elements = [self.item.input, self.item.accumulator, self.out];
        let input = bound(input, self.item.input, self.width, &input_dimensions, edge);
        let out = bound(out, self.out, out_width, &out_dimensions, Boundary::Zero);
        let lanes = self.plane_width as usize;
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            match self.microkernel {
                TestedMicrokernel::UnitDirect => unit_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    UnitDirectReduceMicrokernel::new(self.item, LANE_SLOTS),
                    elements,
                ),
                TestedMicrokernel::PlaneShuffle => plane_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    PlaneShuffleReduceMicrokernel::new(self.item, lanes, LANE_SLOTS),
                    elements,
                ),
                TestedMicrokernel::CubeStaged { planes } => cube_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    CubeStagedReduceMicrokernel::new(self.item, lanes, planes, LANE_SLOTS),
                    elements,
                ),
                TestedMicrokernel::HypercubeLastArrival { .. } => {
                    unreachable!("a hypercube splits its rows apart")
                }
            }
        }
        Ok(())
    }

    /// Every row split into `parts` equal parts, a cube of `planes` planes a part: the input laid
    /// `input[o, p, r]` along the reduced dimension, `input[p, r, o]` across it. The partials and
    /// the cubes' counters are the launch's own.
    fn launch_split(
        &self,
        client: &Client,
        input: TensorArg,
        out: TensorArg,
        planes: usize,
        parts: usize,
    ) -> Result<(), Refusal> {
        let cells = self.item.operation.written_per_row();
        let part = self.reduced / parts;
        assert_eq!(part * parts, self.reduced, "the parts split the row evenly");
        // A partial is `cells` lines; along, each line holds `width` lanes of one row.
        let (
            owned,
            input_dimensions,
            out_dimensions,
            partial_dimensions,
            partial_lines,
            part_lines,
        ) = match self.item.layout {
            ReduceLayout::Along => (
                1,
                [ROWS, PARTS, REDUCED],
                [ROWS, SLOTS],
                [ROWS, PARTIAL_LINES],
                parts * cells * self.width,
                part / self.width,
            ),
            ReduceLayout::Across => (
                self.width,
                [PARTS, REDUCED, ROWS],
                [SLOTS, ROWS],
                [PARTIAL_LINES, ROWS],
                parts * cells,
                part,
            ),
        };
        let space = Space::new(&[
            (ROWS, self.rows),
            (REDUCED, part),
            (SLOTS, cells),
            (PARTS, parts),
            (PARTIAL_LINES, partial_lines),
        ]);
        let leaf = Levels::leaf(&[
            (ROWS, owned),
            (REDUCED, part),
            (SLOTS, cells),
            (PARTS, 1),
            (PARTIAL_LINES, partial_lines),
        ]);
        let partitioning = Partitioning::new(space.clone(), leaf.cubes(&[ROWS, PARTS]).build());
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: CubeDim::new_2d(self.plane_width, planes as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let partial_shape = match self.item.layout {
            ReduceLayout::Along => [self.rows, partial_lines],
            ReduceLayout::Across => [partial_lines, self.rows],
        };
        let index = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::zeros(client, partial_shape, self.item.accumulator);
        let arrivals = TensorHandle::zeros(client, [self.rows], index);
        // Past its edge the input reads the identity of the operation's edge monoid; the outputs and
        // the partials are written, never read past theirs.
        let bound = |arg, element, width, dimensions: &[_], boundary| {
            TileTensorLaunch::new(
                arg,
                element,
                width,
                TileSpec::direct(dimensions).boundary(BoundaryPolicy::Every(boundary)),
            )
        };
        let edge = Boundary::Identity(self.item.operation.edge_monoid());
        let microkernel =
            HypercubeLastArrivalReduceMicrokernel::new(CubeStagedReduceMicrokernel::new(
                self.item,
                self.plane_width as usize,
                planes,
                LANE_SLOTS,
            ));
        // The partials' indices, only where the carrier keeps any.
        // SAFETY: every operand but the counters is bound through the launcher at the problem's
        // own extents, and the kernel indexes no buffer but through its tiles and the counters,
        // one a row, each owner's at its first row.
        match self.item.operation.carrier().carries_indices() {
            true => unsafe {
                hypercube_indexed_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    bound(input, self.item.input, self.width, &input_dimensions, edge),
                    bound(out, self.out, owned, &out_dimensions, Boundary::Zero),
                    bound(
                        partials.binding().into_tensor_arg(),
                        self.item.accumulator,
                        self.width,
                        &partial_dimensions,
                        Boundary::Zero,
                    ),
                    bound(
                        TensorHandle::zeros(client, partial_shape, index)
                            .binding()
                            .into_tensor_arg(),
                        index,
                        self.width,
                        &partial_dimensions,
                        Boundary::Zero,
                    ),
                    arrivals.binding().into_tensor_arg().into_buffer_arg(),
                    launcher.partitioning_arg(),
                    parts,
                    part_lines,
                    self.reduced,
                    microkernel,
                    [self.item.input, self.item.accumulator, self.out],
                )
            },
            false => unsafe {
                hypercube_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    bound(input, self.item.input, self.width, &input_dimensions, edge),
                    bound(out, self.out, owned, &out_dimensions, Boundary::Zero),
                    bound(
                        partials.binding().into_tensor_arg(),
                        self.item.accumulator,
                        self.width,
                        &partial_dimensions,
                        Boundary::Zero,
                    ),
                    arrivals.binding().into_tensor_arg().into_buffer_arg(),
                    launcher.partitioning_arg(),
                    parts,
                    part_lines,
                    self.reduced,
                    microkernel,
                    [self.item.input, self.item.accumulator, self.out],
                )
            },
        };
        Ok(())
    }
}

/// Every unit's row, reduced by the unit alone.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn unit_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);

        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &row,
            &mut row_out,
            &ComptimeOption::new_None(),
        );
    }
}

/// Every plane's row, reduced by the plane.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn plane_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);

    for plane in &partitioning {
        let mut row_out = o.at(&plane);
        let row = x.at(&plane);

        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &row,
            &mut row_out,
            &ComptimeOption::new_None(),
        );
    }
}

/// Every cube's row, reduced by the cube.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn cube_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);

    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        let row = x.at(&cube);

        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &row,
            &mut row_out,
            &ComptimeOption::new_None(),
        );
    }
}

/// Every row's part, reduced by its cube into a carrier of values alone; the row's last cube
/// merges the parts.
///
/// Its operands' leaves:
/// - `input`: the rows, each split into `parts` equal parts.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
/// - `partials`: every part's carrier of every row, in the accumulator's element.
/// - `arrivals`: one `u32` counter a row, at zero.
#[cube(launch_unchecked)]
fn hypercube_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partials: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] parts: usize,
    #[comptime] part_lines: usize,
    #[comptime] values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    let p = partials.tile(&partitioning);

    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        let row = x.at(&cube);
        let mut slots = p.at(&cube);

        let counter = cube.origin(ROWS);
        let part = cube.origin(PARTS);
        let first_line = part * part_lines;

        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &row,
            &mut row_out,
            &ComptimeOption::new_None(),
            &mut slots,
            &ComptimeOption::new_None(),
            arrivals,
            counter,
            part,
            parts,
            first_line,
            values,
        );
    }
}

/// Every row's part, reduced by its cube into a carrier of values and their indices, whose
/// partials keep both; the row's last cube merges the parts.
///
/// Its operands' leaves:
/// - `input`: the rows, each split into `parts` equal parts.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
/// - `partials`: every part's carrier of every row, in the accumulator's element.
/// - `partial_indices`: the index of every value `partials` keeps, in `u32`.
/// - `arrivals`: one `u32` counter a row, at zero.
#[cube(launch_unchecked)]
fn hypercube_indexed_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partials: &TileTensor<'_>,
    partial_indices: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] parts: usize,
    #[comptime] part_lines: usize,
    #[comptime] values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    let p = partials.tile(&partitioning);
    let pi = partial_indices.tile(&partitioning);

    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        let row = x.at(&cube);
        let mut slots = p.at(&cube);

        let counter = cube.origin(ROWS);
        let part = cube.origin(PARTS);
        let first_line = part * part_lines;

        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &row,
            &mut row_out,
            &ComptimeOption::new_None(),
            &mut slots,
            &ComptimeOption::new_Some(pi.at(&cube)),
            arrivals,
            counter,
            part,
            parts,
            first_line,
            values,
        );
    }
}
