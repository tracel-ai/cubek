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

/// The rows one launch reduces: how many of them, how many values long, how many of them a line holds
/// and how many units the planes that read them do, and the element they are written in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LaunchedRows {
    pub item: ReduceWorkItem,
    pub row_count: usize,
    pub value_count: usize,
    pub line_lane_count: usize,
    pub plane_lane_count: usize,
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
            LineDirection::Across => self.line_lane_count,
        }
    }

    /// The lanes of a line that hold one row: all of them along the reduced dimension, one across it.
    pub(crate) fn row_lanes(&self) -> usize {
        match self.direction() {
            LineDirection::Along => self.line_lane_count,
            LineDirection::Across => 1,
        }
    }

    /// Every row and every position along it.
    pub(crate) fn space(&self) -> Space {
        Space::new(&[
            (BATCH, self.row_count),
            (REDUCED, self.value_count),
            (SLOTS, self.cells()),
        ])
    }

    /// An owner's box of rows, one line wide.
    pub(crate) fn owner(&self) -> [(Dimension, usize); 2] {
        [(BATCH, self.owned()), (SLOTS, self.cells())]
    }

    /// The input's dimensions, outermost first.
    pub(crate) fn input_dimensions(&self) -> [Dimension; 2] {
        match self.direction() {
            LineDirection::Along => [BATCH, REDUCED],
            LineDirection::Across => [REDUCED, BATCH],
        }
    }

    /// The output's dimensions, outermost first.
    pub(crate) fn out_dimensions(&self) -> [Dimension; 2] {
        match self.direction() {
            LineDirection::Along => [BATCH, SLOTS],
            LineDirection::Across => [SLOTS, BATCH],
        }
    }

    /// The levels of a launch whose units each own a box of rows, the whole of each.
    pub(crate) fn unit_levels(&self) -> Levels {
        self.whole_rows().launch_units(
            &[BATCH],
            UnitCube {
                plane_width: self.plane_lane_count,
                planes: 1,
            },
        )
    }

    /// A leaf of an owner's rows, the whole of each.
    pub(crate) fn whole_rows(&self) -> Levels {
        Levels::leaf(&[
            (BATCH, self.owned()),
            (REDUCED, self.value_count),
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
    pub out: TensorBinding,
    pub rows: LaunchedRows,
}

impl LaunchedRows {
    /// `input` bound over `dimensions`. Past its edge it reads the identity of the operation's edge
    /// monoid.
    pub(crate) fn bound_input(
        &self,
        input: TensorArg,
        dimensions: &[Dimension],
    ) -> TileTensorLaunch {
        let edge = Boundary::Identity(self.item.operation.edge_monoid());
        bound(
            input,
            self.item.input,
            self.line_lane_count,
            dimensions,
            edge,
        )
    }

    /// `out` bound over the output's dimensions, written only within its own edge.
    pub(crate) fn bound_out(&self, out: TensorBinding) -> TileTensorLaunch {
        let dimensions = self.out_dimensions();
        bound(
            out.into_tensor_arg(),
            self.out,
            self.owned(),
            &dimensions,
            Boundary::Zero,
        )
    }
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
