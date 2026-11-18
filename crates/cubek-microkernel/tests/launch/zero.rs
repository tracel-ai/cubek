//! The zero microkernels on one launch: `out[o, c]`, every row's cells zeroed by its owner.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{
    Axis, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::zero::unit::UnitZeroMicrokernel;

/// The rows of the output, a plane each.
const O: Axis = Axis(0);
/// The cells of a row.
const C: Axis = Axis(1);

/// `out`, `rows` rows of `cells` cells in `element`, row-major, zeroed by the first unit of each
/// row's plane.
///
/// # Errors
///
/// Why the tile DSL cannot bind the launch.
pub fn launch_unit_zero(
    client: &Client,
    out: TensorArg,
    element: ElemType,
    rows: usize,
    cells: usize,
    plane_width: u32,
) -> Result<(), Refusal> {
    let space = Space::new(&[(O, rows), (C, cells)]);
    let partitioning = Partitioning::new(
        space.clone(),
        Levels::leaf(&[(O, 1), (C, cells)])
            .launch_planes(&[O], PlaneOrder::new(&[(O, 1)], CubeOrder::RowMajor))
            .build(),
    );
    let grid = Grid::Stated {
        cube_count: partitioning.cube_count(),
        cube_dim: partitioning.cube_dim(plane_width),
    };
    let launcher = Launcher::new(client, partitioning, &space, grid)?;
    let out = TileTensorLaunch::new(out, element, 1, TileSpec::direct(&[O, C]));
    // SAFETY: the output is bound through the launcher at its own extents, and the kernel indexes
    // no buffer but through its tile.
    unsafe {
        unit_zero_kernel::launch_unchecked(
            client,
            launcher.cube_count(),
            launcher.cube_dim(),
            out,
            launcher.partitioning_arg(),
            UnitZeroMicrokernel {},
        );
    }
    Ok(())
}

/// Every plane's row, zeroed by its first unit alone.
#[cube(launch_unchecked)]
fn unit_zero_kernel(
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitZeroMicrokernel,
) {
    let o = out.tile(&partitioning);
    for plane in &partitioning {
        if UNIT_POS_PLANE == 0 {
            let mut cells = o.at(&plane);
            microkernel.runtime().execute(&mut cells);
        }
    }
}
