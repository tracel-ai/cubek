//! The zero microkernels on one launch: `out[o, c]`, every row's cells zeroed by its owner.

use cubecl::prelude::*;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Decline, Grid};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{
    Dimension, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::zero::unit::UnitDirectZeroMicrokernel;

/// The rows of the output, a unit each.
const ROWS: Dimension = Dimension(0);
/// The cells of a row.
const CELLS: Dimension = Dimension(1);

/// `out`, `rows` rows of `cells` cells in `element`, row-major, each row zeroed by a unit of its
/// own, a cube of one plane of `plane_width` units holding as many rows.
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
) -> Result<(), Decline> {
    let space = Space::new(&[(ROWS, rows), (CELLS, cells)]);
    let partitioning = Partitioning::new(
        space.clone(),
        Levels::leaf(&[(ROWS, 1), (CELLS, cells)])
            .launch_units(
                &[ROWS],
                UnitCube {
                    plane_width: plane_width as usize,
                    planes: 1,
                },
            )
            .build(),
    );
    let grid = Grid::Stated {
        cube_count: partitioning.cube_count(),
        cube_dim: partitioning.cube_dim(plane_width),
    };
    let launcher = Launcher::new(client, partitioning, &space, grid)?;
    // A cube's units past the last row write nothing: their writes are masked.
    let out = TileTensorLaunch::new(
        out,
        element,
        1,
        TileSpec::direct(&[ROWS, CELLS]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
    );
    // SAFETY: the output is bound through the launcher, which masks the writes of the units past
    // its last row, and the kernel indexes no buffer but through its tile.
    unsafe {
        unit_zero_kernel::launch_unchecked(
            client,
            launcher.cube_count(),
            launcher.cube_dim(),
            out,
            launcher.partitioning_arg(),
            UnitDirectZeroMicrokernel {},
        );
    }
    Ok(())
}

/// Every unit's row, zeroed by the unit.
///
/// Its operands' leaves:
/// - `out`: one tensor of `rows` rows of `cells` cells, row-major.
#[cube(launch_unchecked)]
fn unit_zero_kernel(
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectZeroMicrokernel,
) {
    let o = out.tile(&partitioning);
    for unit in &partitioning {
        let mut cells = o.at(&unit);
        microkernel.runtime().execute(&mut cells);
    }
}
