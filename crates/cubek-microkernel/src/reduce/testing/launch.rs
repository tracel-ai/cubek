use crate::reduce::cube::{CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings};
use crate::reduce::plane::{PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings};
use crate::reduce::testing::LineDirection;
use crate::reduce::{BATCH, REDUCED, ReduceWorkItem, SLOTS};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Decline};
use cubecl::tile::space::{CubeOrder, PlaneOrder, UnitCube};
use cubecl::tile::{Dimension, Launcher, Levels, Space, TileSpec, TileTensorLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The top-k slots a thread keeps, the first the microkernels' settings list.
pub(crate) const LANE_SLOT_COUNT: usize = crate::reduce::TOP_K_LANE_SLOT_COUNTS[0];

/// The rows one launch reduces: `rows` of them, `reduced` values long, read `width` values a line by
/// planes of `lanes` units, written in `out`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LaunchedRows {
    pub item: ReduceWorkItem,
    pub rows: usize,
    pub reduced: usize,
    pub width: usize,
    pub lanes: usize,
    pub out: ElemType,
}

impl LaunchedRows {
    /// Cells one row writes.
    pub(crate) fn cells(&self) -> usize {
        self.item.operation.written_per_row()
    }

    /// Which way the lines run.
    pub(crate) fn direction(&self) -> LineDirection {
        LineDirection::from_lines_along(self.item.lines_along)
    }

    /// Rows an owner holds: across, a line's worth of them side by side.
    pub(crate) fn owned(&self) -> usize {
        match self.direction() {
            LineDirection::Along => 1,
            LineDirection::Across => self.width,
        }
    }

    /// The lanes of a line that hold one row: all of them along the reduced dimension, one across it.
    pub(crate) fn row_lanes(&self) -> usize {
        match self.direction() {
            LineDirection::Along => self.width,
            LineDirection::Across => 1,
        }
    }

    /// Every row and every position along it.
    pub(crate) fn space(&self) -> Space {
        Space::new(&[
            (BATCH, self.rows),
            (REDUCED, self.reduced),
            (SLOTS, self.cells()),
        ])
    }

    /// An owner's box of rows, one line wide.
    pub(crate) fn owner(&self) -> [(Dimension, usize); 2] {
        [(BATCH, self.owned()), (SLOTS, self.cells())]
    }

    /// The input's dimensions, outermost first.
    pub(crate) fn input_dimensions(&self) -> Vec<Dimension> {
        match self.direction() {
            LineDirection::Along => vec![BATCH, REDUCED],
            LineDirection::Across => vec![REDUCED, BATCH],
        }
    }

    /// The output's dimensions, outermost first.
    pub(crate) fn out_dimensions(&self) -> Vec<Dimension> {
        match self.direction() {
            LineDirection::Along => vec![BATCH, SLOTS],
            LineDirection::Across => vec![SLOTS, BATCH],
        }
    }

    /// The levels of a launch whose units each own a box of rows, the whole of each.
    pub(crate) fn unit_levels(&self) -> Levels {
        self.whole_rows().launch_units(
            &[BATCH],
            UnitCube {
                plane_width: self.lanes,
                planes: 1,
            },
        )
    }

    /// A leaf of an owner's rows, the whole of each.
    pub(crate) fn whole_rows(&self) -> Levels {
        Levels::leaf(&[
            (BATCH, self.owned()),
            (REDUCED, self.reduced),
            (SLOTS, self.cells()),
        ])
    }
}

/// `levels` with a plane a cube, its planes dealt rows.
pub(crate) fn a_plane_a_cube(levels: Levels) -> Levels {
    let order = PlaneOrder::new(&[(BATCH, 1)], CubeOrder::RowMajor);
    levels.launch_planes(&[BATCH], order)
}

/// `arg` launched in `element`, `width` wide over `dimensions`, reading `boundary` past its edge.
pub(crate) fn bound(
    arg: TensorArg,
    element: ElemType,
    width: usize,
    dimensions: &[Dimension],
    boundary: Boundary,
) -> TileTensorLaunch {
    TileTensorLaunch::new(
        arg,
        element,
        width,
        TileSpec::direct(dimensions).boundary(BoundaryPolicy::Every(boundary)),
    )
}

/// A launch of one microkernel, its operands bound.
pub(crate) struct ComponentLaunch<'a> {
    pub client: &'a Client,
    pub launcher: Launcher,
    pub input: TileTensorLaunch,
    pub out: TileTensorLaunch,
    pub rows: LaunchedRows,
}

impl ComponentLaunch<'_> {
    /// The input's, the accumulator's and the output's elements.
    pub(crate) fn elements(&self) -> [ElemType; 3] {
        [
            self.rows.item.input,
            self.rows.item.accumulator,
            self.rows.out,
        ]
    }
}

/// The units of a plane the plane's microkernels reduce `item` with on `hw`.
///
/// # Errors
///
/// Why the device commits to no plane width.
pub(crate) fn plane_lanes(hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
    let settings = PlaneShuffleReduceMicrokernelSettings {
        lane_slot_count: LANE_SLOT_COUNT,
    };
    PlaneShuffleReduceMicrokernel::best(hw, item, &settings).map(|plane| plane.lanes())
}

/// The units of a plane a cube of `planes` planes reduces `item` with on `hw`.
///
/// # Errors
///
/// Why the device runs no such cube.
pub(crate) fn cube_lanes(
    hw: &Hardware,
    item: &ReduceWorkItem,
    planes: usize,
) -> Result<usize, Decline> {
    let settings = CubeStagedReduceMicrokernelSettings {
        plane_count: planes,
        lane_slot_count: LANE_SLOT_COUNT,
    };
    CubeStagedReduceMicrokernel::best(hw, item, &settings).map(|cube| cube.lanes())
}
