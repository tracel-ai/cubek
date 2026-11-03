//! The reduce microkernels on one launch: `out[o, s]` the `s`-th cell row `o` writes (one, or `k`
//! for a top-k). Along the reduced axis, `input[o, r]`, each row its owner's; across it,
//! `input[r, o]` and `out[s, o]`, each owner `width` rows side by side, a row a lane.

use crate::reduce::cube::{
    CubeReduceMicrokernel, CubeReduceMicrokernelSettings, cube_reduce_microkernel,
};
use crate::reduce::plane::{
    PlaneReduceMicrokernel, PlaneReduceMicrokernelSettings, plane_reduce_microkernel,
};
use crate::reduce::unit::{UnitReduceMicrokernel, unit_reduce_microkernel};
use crate::reduce::{O, R, ReduceCarrier, ReduceLayout, ReduceWorkItem};
use cubecl::prelude::*;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{
    Axis, Indexed, Launcher, Levels, Partitioning, Space, TileArg, TileArgLaunch, TileSpec, TopK,
};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The cells a row writes: one, or `k` for a top-k.
const S: Axis = Axis(2);

/// Which microkernel each row's plane runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceScope {
    /// The plane's first unit alone.
    Unit,
    /// The whole plane.
    Plane,
    /// A cube of `planes` planes.
    Cube { planes: usize },
}

/// A reduce microkernel ready to launch on `rows` rows, each `item.reduced` values long and
/// contiguous, read `width` values a line.
#[derive(Debug, Clone)]
pub struct ReduceLaunch {
    pub item: ReduceWorkItem,
    pub scope: ReduceScope,
    pub rows: usize,
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
        width: usize,
        out: ElemType,
    ) -> Result<Self, Refusal> {
        let plane_width = match scope {
            ReduceScope::Unit => hw.plane_dim().unwrap_or(1),
            ReduceScope::Plane => {
                PlaneReduceMicrokernel::best(hw, &item, &PlaneReduceMicrokernelSettings)?.lanes
                    as u32
            }
            ReduceScope::Cube { planes } => {
                CubeReduceMicrokernel::best(hw, &item, &CubeReduceMicrokernelSettings { planes })?
                    .lanes as u32
            }
        };
        Ok(Self {
            item,
            scope,
            rows,
            width,
            out,
            plane_width,
        })
    }

    /// `out[o, s]` from `input[o, r]` along the reduced axis, `out[s, o]` from `input[r, o]`
    /// across it, all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(&self, client: &Client, input: TensorArg, out: TensorArg) -> Result<(), Refusal> {
        let cells = self.item.operation.written_per_row();
        let space = Space::new(&[(O, self.rows), (R, self.item.reduced), (S, cells)]);
        // Across, an owner holds a line's worth of rows side by side.
        let (owned, input_axes, out_axes, out_width) = match self.item.layout {
            ReduceLayout::Along => (1, [O, R], [O, S], 1),
            ReduceLayout::Across => (self.width, [R, O], [S, O], self.width),
        };
        let leaf = Levels::leaf(&[(O, owned), (R, self.item.reduced), (S, cells)]);
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
        };
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim,
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |arg, element, width, axes: &[_]| {
            TileArgLaunch::new(
                arg,
                element,
                width,
                TileSpec::direct(axes).boundary(BoundaryPolicy::Every(Boundary::Zero)),
            )
        };
        let elements = [self.item.input, self.item.accumulator, self.out];
        let input = bound(input, self.item.input, self.width, &input_axes);
        let out = bound(out, self.out, out_width, &out_axes);
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
                    UnitReduceMicrokernel { item: self.item },
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
                    PlaneReduceMicrokernel {
                        item: self.item,
                        lanes,
                    },
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
                    CubeReduceMicrokernel {
                        item: self.item,
                        lanes,
                        planes,
                    },
                    elements,
                ),
            }
        }
        Ok(())
    }
}

/// Every plane's row, reduced by its first unit alone.
#[cube(launch_unchecked)]
fn unit_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileArg<'_>,
    out: &TileArg<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitReduceMicrokernel,
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
            match comptime!(microkernel.item.operation.carrier()) {
                ReduceCarrier::Values => {
                    unit_reduce_microkernel::<In, Acc, Out, Out, N, Vector<Acc, N>>(
                        &row,
                        &mut row_out,
                        &mut unwritten,
                        microkernel,
                    )
                }
                ReduceCarrier::Indexed => {
                    unit_reduce_microkernel::<In, Acc, Out, Out, N, Indexed<Acc, N>>(
                        &row,
                        &mut row_out,
                        &mut unwritten,
                        microkernel,
                    )
                }
                ReduceCarrier::TopK { k } => {
                    let size!(K) = k;
                    unit_reduce_microkernel::<In, Acc, Out, Out, N, TopK<Acc, N, K>>(
                        &row,
                        &mut row_out,
                        &mut unwritten,
                        microkernel,
                    )
                }
            }
        }
    }
}

/// Every plane's row, reduced by the plane.
#[cube(launch_unchecked)]
fn plane_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileArg<'_>,
    out: &TileArg<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    for plane in &partitioning {
        let mut row_out = o.at(&plane);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&plane);
        let row = x.at(&plane);
        match comptime!(microkernel.item.operation.carrier()) {
            ReduceCarrier::Values => {
                plane_reduce_microkernel::<In, Acc, Out, Out, N, Vector<Acc, N>>(
                    &row,
                    &mut row_out,
                    &mut unwritten,
                    microkernel,
                )
            }
            ReduceCarrier::Indexed => {
                plane_reduce_microkernel::<In, Acc, Out, Out, N, Indexed<Acc, N>>(
                    &row,
                    &mut row_out,
                    &mut unwritten,
                    microkernel,
                )
            }
            ReduceCarrier::TopK { k } => {
                let size!(K) = k;
                plane_reduce_microkernel::<In, Acc, Out, Out, N, TopK<Acc, N, K>>(
                    &row,
                    &mut row_out,
                    &mut unwritten,
                    microkernel,
                )
            }
        }
    }
}

/// Every cube's row, reduced by the cube.
#[cube(launch_unchecked)]
fn cube_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileArg<'_>,
    out: &TileArg<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let row = x.at(&cube);
        match comptime!(microkernel.item.operation.carrier()) {
            ReduceCarrier::Values => {
                cube_reduce_microkernel::<In, Acc, Out, Out, N, Vector<Acc, N>>(
                    &row,
                    &mut row_out,
                    &mut unwritten,
                    microkernel,
                )
            }
            ReduceCarrier::Indexed => {
                cube_reduce_microkernel::<In, Acc, Out, Out, N, Indexed<Acc, N>>(
                    &row,
                    &mut row_out,
                    &mut unwritten,
                    microkernel,
                )
            }
            ReduceCarrier::TopK { k } => {
                let size!(K) = k;
                cube_reduce_microkernel::<In, Acc, Out, Out, N, TopK<Acc, N, K>>(
                    &row,
                    &mut row_out,
                    &mut unwritten,
                    microkernel,
                )
            }
        }
    }
}
