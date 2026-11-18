//! The reduce microkernels on one launch: `out[o, s]` the `s`-th cell row `o` writes (one, or `k`
//! for a top-k). Along the reduced dimension, `input[o, r]`, each row its owner's; across it,
//! `input[r, o]` and `out[s, o]`, each owner `width` rows side by side, a row a lane. A hypercube
//! splits the reduced dimension into parts, `input[o, p, r]` or `input[p, r, o]`, a cube a part.

use cubecl::ir::UIntKind;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
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
use cubek_microkernel::reduce::{O, R, ReduceLayout, ReduceWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The cells a row writes: one, or `k` for a top-k.
const S: Dimension = Dimension(2);
/// The parts a hypercube splits the reduced dimension into, a cube a part.
const P: Dimension = Dimension(3);
/// The lines of a row's partials, every part's, which only they run along.
const Q: Dimension = Dimension(4);

/// Which microkernel each row's plane runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceScope {
    /// The plane's first unit alone.
    Unit,
    /// The whole plane.
    Plane,
    /// A cube of `planes` planes.
    Cube { planes: usize },
    /// `parts` cubes of `planes` planes, each reducing its part of the row, the last to finish
    /// merging the parts.
    Hypercube { planes: usize, parts: usize },
}

/// A reduce microkernel ready to launch on `rows` rows, each `reduced` values long and
/// contiguous, read `width` values a line.
#[derive(Debug, Clone)]
pub struct ReduceLaunch {
    pub item: ReduceWorkItem,
    pub scope: ReduceScope,
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
        scope: ReduceScope,
        rows: usize,
        reduced: usize,
        width: usize,
        out: ElemType,
    ) -> Result<Self, Refusal> {
        let plane_width = match scope {
            ReduceScope::Unit => hw.plane_dim().unwrap_or(1),
            ReduceScope::Plane => PlaneShuffleReduceMicrokernel::best(
                hw,
                &item,
                &PlaneShuffleReduceMicrokernelSettings,
            )?
            .lanes() as u32,
            ReduceScope::Cube { planes } => CubeStagedReduceMicrokernel::best(
                hw,
                &item,
                &CubeStagedReduceMicrokernelSettings { planes },
            )?
            .lanes() as u32,
            ReduceScope::Hypercube { planes, .. } => {
                let settings = CubeStagedReduceMicrokernelSettings { planes };
                HypercubeLastArrivalReduceMicrokernel::best(hw, &item, &settings)?
                    .part()
                    .lanes() as u32
            }
        };
        Ok(Self {
            item,
            scope,
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
        match self.scope {
            ReduceScope::Hypercube { planes, parts } => {
                self.launch_split(client, input, out, planes, parts)
            }
            ReduceScope::Unit | ReduceScope::Plane | ReduceScope::Cube { .. } => {
                self.launch_whole(client, input, out)
            }
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
        let space = Space::new(&[(O, self.rows), (R, self.reduced), (S, cells)]);
        // Across, an owner holds a line's worth of rows side by side.
        let (owned, input_dimensions, out_dimensions, out_width) = match self.item.layout {
            ReduceLayout::Along => (1, [O, R], [O, S], 1),
            ReduceLayout::Across => (self.width, [R, O], [S, O], self.width),
        };
        let leaf = Levels::leaf(&[(O, owned), (R, self.reduced), (S, cells)]);
        let (partitioning, cube_dim) = match self.scope {
            ReduceScope::Unit | ReduceScope::Plane => {
                let partitioning = Partitioning::new(
                    space.clone(),
                    leaf.launch_planes(&[O], PlaneOrder::new(&[(O, 1)], CubeOrder::RowMajor))
                        .build(),
                );
                let cube_dim = partitioning.cube_dim(self.plane_width);
                (partitioning, cube_dim)
            }
            ReduceScope::Cube { planes } => (
                Partitioning::new(space.clone(), leaf.cubes(&[O]).build()),
                CubeDim::new_2d(self.plane_width, planes as u32),
            ),
            ReduceScope::Hypercube { .. } => unreachable!("a hypercube splits its rows apart"),
        };
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim,
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |arg, element, width, dimensions: &[_]| {
            TileTensorLaunch::new(
                arg,
                element,
                width,
                TileSpec::direct(dimensions).boundary(BoundaryPolicy::Every(Boundary::Zero)),
            )
        };
        let elements = [self.item.input, self.item.accumulator, self.out];
        let input = bound(input, self.item.input, self.width, &input_dimensions);
        let out = bound(out, self.out, out_width, &out_dimensions);
        let lanes = self.plane_width as usize;
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            match self.scope {
                ReduceScope::Unit => unit_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    UnitDirectReduceMicrokernel::new(self.item),
                    elements,
                ),
                ReduceScope::Plane => plane_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    PlaneShuffleReduceMicrokernel::new(self.item, lanes),
                    elements,
                ),
                ReduceScope::Cube { planes } => cube_reduce_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    CubeStagedReduceMicrokernel::new(self.item, lanes, planes),
                    elements,
                ),
                ReduceScope::Hypercube { .. } => unreachable!("a hypercube splits its rows apart"),
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
                [O, P, R],
                [O, S],
                [O, Q],
                parts * cells * self.width,
                part / self.width,
            ),
            ReduceLayout::Across => (self.width, [P, R, O], [S, O], [Q, O], parts * cells, part),
        };
        let space = Space::new(&[
            (O, self.rows),
            (R, part),
            (S, cells),
            (P, parts),
            (Q, partial_lines),
        ]);
        let leaf = Levels::leaf(&[
            (O, owned),
            (R, part),
            (S, cells),
            (P, 1),
            (Q, partial_lines),
        ]);
        let partitioning = Partitioning::new(space.clone(), leaf.cubes(&[O, P]).build());
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
        let partial_indices = TensorHandle::zeros(client, partial_shape, index);
        let arrivals = TensorHandle::zeros(client, [self.rows], index);
        let bound = |arg, element, width, dimensions: &[_]| {
            TileTensorLaunch::new(
                arg,
                element,
                width,
                TileSpec::direct(dimensions).boundary(BoundaryPolicy::Every(Boundary::Zero)),
            )
        };
        let microkernel = HypercubeLastArrivalReduceMicrokernel::new(
            CubeStagedReduceMicrokernel::new(self.item, self.plane_width as usize, planes),
        );
        // SAFETY: every operand but the counters is bound through the launcher at the problem's
        // own extents, and the kernel indexes no buffer but through its tiles and the counters,
        // one a row, each owner's at its first row.
        unsafe {
            hypercube_reduce_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                self.width,
                bound(input, self.item.input, self.width, &input_dimensions),
                bound(out, self.out, owned, &out_dimensions),
                bound(
                    partials.binding().into_tensor_arg(),
                    self.item.accumulator,
                    self.width,
                    &partial_dimensions,
                ),
                bound(
                    partial_indices.binding().into_tensor_arg(),
                    index,
                    self.width,
                    &partial_dimensions,
                ),
                arrivals.binding().into_tensor_arg().into_buffer_arg(),
                launcher.partitioning_arg(),
                parts,
                part_lines,
                self.reduced,
                microkernel,
                [self.item.input, self.item.accumulator, self.out],
            )
        };
        Ok(())
    }
}

/// Every plane's row, reduced by its first unit alone.
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

    for plane in &partitioning {
        if UNIT_POS_PLANE == 0 {
            let mut row_out = o.at(&plane);
            // One output: the indices are never written, so they get the output's own window.
            let mut unwritten = o.at(&plane);
            let row = x.at(&plane);

            microkernel.runtime().execute::<In, Acc, Out, Out, N>(
                &row,
                &mut row_out,
                &mut unwritten,
            );
        }
    }
}

/// Every plane's row, reduced by the plane.
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
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&plane);
        let row = x.at(&plane);

        microkernel
            .runtime()
            .execute::<In, Acc, Out, Out, N>(&row, &mut row_out, &mut unwritten);
    }
}

/// Every cube's row, reduced by the cube.
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
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let row = x.at(&cube);

        microkernel
            .runtime()
            .execute::<In, Acc, Out, Out, N>(&row, &mut row_out, &mut unwritten);
    }
}

/// Every row's part, reduced by its cube; the row's last cube merges the parts.
#[cube(launch_unchecked)]
fn hypercube_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
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
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let row = x.at(&cube);
        let mut slots = p.at(&cube);
        let mut slot_indices = pi.at(&cube);

        let counter = cube.origin(O);
        let part = cube.origin(P);
        let first_line = part * part_lines;

        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &row,
            &mut row_out,
            &mut unwritten,
            &mut slots,
            &mut slot_indices,
            arrivals,
            counter,
            part,
            parts,
            first_line,
            values,
        );
    }
}
