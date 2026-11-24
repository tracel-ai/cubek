//! The reduce microkernels on one launch: `out[BATCH, SLOTS]`, the cells each row writes (one, or
//! `k` for a top-k). Along the reduced dimension, `input[BATCH, REDUCED]`, each row its owner's;
//! across it, `input[REDUCED, BATCH]` and `out[SLOTS, BATCH]`, each owner `width` rows side by side,
//! a row a lane. A hypercube splits the reduced dimension into parts,
//! `input[BATCH, PARTS, REDUCED]` or `input[PARTS, REDUCED, BATCH]`, a cube a part.

use cubecl::ir::UIntKind;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Decline, Grid};
use cubecl::tile::space::{CubeOrder, PlaneOrder, UnitCube};
use cubecl::tile::{
    Dimension, Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::reduce::cube::{
    CubeColumnsReduceMicrokernel, CubeLineReduceMicrokernel, CubePartialReduceMicrokernel,
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings, CubeStridedReduceMicrokernel,
};
use cubek_microkernel::reduce::hypercube::HypercubeLastArrivalReduceMicrokernel;
use cubek_microkernel::reduce::plane::{
    PlaneColumnsReduceMicrokernel, PlaneLineReduceMicrokernel, PlaneShuffleReduceMicrokernel,
    PlaneShuffleReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::read::{
    UnitPartialsReduceMicrokernel, UnitStridedReduceMicrokernel, UnitWalkReduceMicrokernel,
};
use cubek_microkernel::reduce::unit::UnitDirectReduceMicrokernel;
use cubek_microkernel::reduce::write::{
    UnitCarrierWriteMicrokernel, UnitLinesWriteMicrokernel, UnitPartialStoreMicrokernel,
    UnitWalkWriteMicrokernel,
};
use cubek_microkernel::reduce::{
    BATCH, IndexedReduceCarrierFamily, REDUCED, ReduceCarrier, ReduceCarrierFamily, ReduceWorkItem,
    SLOTS, TOP_K_LANE_SLOTS, TopKReduceCarrierFamily, TopValuesReduceCarrierFamily,
    ValuesReduceCarrierFamily,
};

/// The lane slots the microkernels' settings list.
const LANE_SLOTS: usize = TOP_K_LANE_SLOTS[0];
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The parts a hypercube splits the reduced dimension into, a cube a part.
const PARTS: Dimension = Dimension(3);
/// The lines of a row's partials, every part's, which only they run along.
const PARTIAL_LINES: Dimension = Dimension(4);

/// Which way a case's lines run against its rows: the buffers a case lays out, and the
/// dimension its work item's lines run along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineDirection {
    /// Along the reduced dimension, a line `width` values of one row.
    Along,
    /// Across it, a line one value of `width` rows side by side.
    Across,
}

impl LineDirection {
    /// The direction of lines that run along `lines_along`.
    pub fn of(lines_along: Dimension) -> Self {
        match lines_along == REDUCED {
            true => Self::Along,
            false => Self::Across,
        }
    }

    /// The dimension lines in this direction run along.
    pub fn dimension(self) -> Dimension {
        match self {
            Self::Along => REDUCED,
            Self::Across => BATCH,
        }
    }
}

/// Which reduce microkernel a case runs, named for the microkernel, and what its scope holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TestedMicrokernel {
    /// The unit microkernel: every unit reduces its own row.
    UnitDirect,
    /// The plane microkernel: the whole plane.
    PlaneShuffle,
    /// The plane's read one line of rows wide, alone.
    PlaneLine,
    /// The plane's read `columns` lines of rows wide, alone.
    PlaneColumns { columns: usize },
    /// The cube microkernel: a cube of `planes` planes.
    CubeStaged { planes: usize },
    /// The cube's read one line of rows wide, alone: a cube of `planes` planes.
    CubeLine { planes: usize },
    /// The cube's read `columns` lines of rows wide, alone: a cube of `planes` planes.
    CubeColumns { planes: usize, columns: usize },
    /// The cube's read into a carrier, alone: a cube of `planes` planes reduces its row and its
    /// first unit's carrier's line is written as it is.
    CubeStrided { planes: usize },
    /// The hypercube microkernel: `parts` cubes of `planes` planes, each reducing its part of the
    /// row, the last to finish merging the parts.
    HypercubeLastArrival { planes: usize, parts: usize },
    /// The write every reduce microkernel ends on, alone: each unit takes its one line of the
    /// input as its carrier of values and writes it.
    UnitCarrierWrite,
    /// The write of a carrier whose lanes are met, alone: each unit takes its one line of the input
    /// as its carrier of values and writes it as wide as it is.
    UnitLinesWrite,
    /// The strided read the plane and cube microkernels share, alone: each unit reads every
    /// `stride`-th line of its own row from the first, and its carrier is written as it is.
    UnitStridedReduce { stride: usize },
    /// The walked read the plane and cube microkernels share, alone: a plane's units lie across
    /// `columns` lines of rows, and each column's first unit's carrier is written as it is. It
    /// reads the values [`stride_on`](Self::stride_on) apart.
    UnitWalkReduce { columns: usize },
    /// The write after a walked read, alone: a plane's units lie across `columns` lines of rows,
    /// each taking its line of the walk's first step as its carrier, and each column's first
    /// unit's is written.
    UnitWalkWrite { columns: usize },
    /// The store of a partial, alone: each unit takes its one line of the input as its carrier and
    /// stores it.
    UnitPartialStore,
    /// The reduce of stored partials, alone: each unit takes every line of its row as a partial.
    UnitPartialsReduce,
    /// The cube's reduce into a partial, alone: a cube of `planes` planes reduces its row and
    /// stores its one partial.
    CubePartial { planes: usize },
}

impl TestedMicrokernel {
    /// How far apart the values it reads of a row lie, on `hw`: every one unless it is one unit's
    /// share of them.
    pub fn stride_on(self, hw: &Hardware) -> usize {
        match self {
            Self::UnitStridedReduce { stride } => stride,
            Self::UnitWalkReduce { columns } => hw.plane_dim().unwrap_or(1) as usize / columns,
            _ => 1,
        }
    }

    /// The values it reads of a row of `len`: the first alone for the walked write, whose carrier
    /// is the walk's first step's.
    pub fn values_of(self, len: usize) -> usize {
        match self {
            Self::UnitWalkWrite { .. } => 1,
            _ => len,
        }
    }
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
    ) -> Result<Self, Decline> {
        let plane_width = match microkernel {
            TestedMicrokernel::UnitDirect
            | TestedMicrokernel::UnitCarrierWrite
            | TestedMicrokernel::UnitLinesWrite
            | TestedMicrokernel::UnitStridedReduce { .. }
            | TestedMicrokernel::UnitPartialStore
            | TestedMicrokernel::UnitPartialsReduce => hw.plane_dim().unwrap_or(1),
            TestedMicrokernel::PlaneShuffle
            | TestedMicrokernel::PlaneLine
            | TestedMicrokernel::PlaneColumns { .. }
            | TestedMicrokernel::UnitWalkReduce { .. }
            | TestedMicrokernel::UnitWalkWrite { .. } => PlaneShuffleReduceMicrokernel::best(
                hw,
                &item,
                &PlaneShuffleReduceMicrokernelSettings {
                    lane_slots: TOP_K_LANE_SLOTS[0],
                },
            )?
            .lanes() as u32,
            TestedMicrokernel::CubeStaged { planes }
            | TestedMicrokernel::CubeLine { planes }
            | TestedMicrokernel::CubeColumns { planes, .. }
            | TestedMicrokernel::CubeStrided { planes }
            | TestedMicrokernel::CubePartial { planes } => CubeStagedReduceMicrokernel::best(
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
    pub fn launch(&self, client: &Client, input: TensorArg, out: TensorArg) -> Result<(), Decline> {
        match self.item.operation.carrier() {
            ReduceCarrier::Values => {
                self.launch_into::<ValuesReduceCarrierFamily>(client, input, out)
            }
            ReduceCarrier::Indexed => {
                self.launch_into::<IndexedReduceCarrierFamily>(client, input, out)
            }
            ReduceCarrier::TopK { .. } => {
                self.launch_into::<TopKReduceCarrierFamily>(client, input, out)
            }
            ReduceCarrier::TopValues { .. } => {
                self.launch_into::<TopValuesReduceCarrierFamily>(client, input, out)
            }
        }
    }

    /// The launch into `F`'s carrier.
    fn launch_into<F: ReduceCarrierFamily>(
        &self,
        client: &Client,
        input: TensorArg,
        out: TensorArg,
    ) -> Result<(), Decline> {
        match self.microkernel {
            TestedMicrokernel::HypercubeLastArrival { planes, parts } => {
                self.launch_split::<F>(client, input, out, planes, parts)
            }
            _ => self.launch_whole::<F>(client, input, out),
        }
    }

    /// Every row reduced by one owner.
    fn launch_whole<F: ReduceCarrierFamily>(
        &self,
        client: &Client,
        input: TensorArg,
        out: TensorArg,
    ) -> Result<(), Decline> {
        let cells = self.item.operation.written_per_row();
        let space = Space::new(&[(BATCH, self.rows), (REDUCED, self.reduced), (SLOTS, cells)]);
        // Across, an owner holds a line's worth of rows side by side.
        let (owned, input_dimensions, out_dimensions, out_width) =
            match LineDirection::of(self.item.lines_along) {
                LineDirection::Along => (1, [BATCH, REDUCED], [BATCH, SLOTS], 1),
                LineDirection::Across => (self.width, [REDUCED, BATCH], [SLOTS, BATCH], self.width),
            };
        let leaf = Levels::leaf(&[(BATCH, owned), (REDUCED, self.reduced), (SLOTS, cells)]);
        let owner = [(BATCH, owned), (SLOTS, cells)];
        let hw = Hardware::from_client(client);
        let lanes = self.plane_width as usize;
        let plane = PlaneShuffleReduceMicrokernel::new(self.item, lanes, LANE_SLOTS)
            .across_rows(&hw, self.width, self.rows);
        let (partitioning, cube_dim) = match self.microkernel {
            TestedMicrokernel::UnitDirect
            | TestedMicrokernel::UnitCarrierWrite
            | TestedMicrokernel::UnitLinesWrite
            | TestedMicrokernel::UnitStridedReduce { .. }
            | TestedMicrokernel::UnitPartialStore
            | TestedMicrokernel::UnitPartialsReduce => {
                let partitioning = Partitioning::new(
                    space.clone(),
                    leaf.launch_units(
                        &[BATCH],
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
                    plane
                        .levels(&owner)
                        .launch_planes(
                            &[BATCH],
                            PlaneOrder::new(&[(BATCH, 1)], CubeOrder::RowMajor),
                        )
                        .build(),
                );
                let cube_dim = partitioning.cube_dim(self.plane_width);
                (partitioning, cube_dim)
            }
            TestedMicrokernel::PlaneLine => {
                let partitioning = Partitioning::new(
                    space.clone(),
                    Levels::leaf(&owner)
                        .launch_planes(
                            &[BATCH],
                            PlaneOrder::new(&[(BATCH, 1)], CubeOrder::RowMajor),
                        )
                        .build(),
                );
                let cube_dim = partitioning.cube_dim(self.plane_width);
                (partitioning, cube_dim)
            }
            TestedMicrokernel::PlaneColumns { columns }
            | TestedMicrokernel::UnitWalkReduce { columns }
            | TestedMicrokernel::UnitWalkWrite { columns } => {
                let partitioning = Partitioning::new(
                    space.clone(),
                    columns_of(&owner, columns, lanes / columns)
                        .walk_every(&[REDUCED])
                        .launch_planes(
                            &[BATCH],
                            PlaneOrder::new(&[(BATCH, 1)], CubeOrder::RowMajor),
                        )
                        .build(),
                );
                let cube_dim = partitioning.cube_dim(self.plane_width);
                (partitioning, cube_dim)
            }
            TestedMicrokernel::CubeLine { planes } => (
                Partitioning::new(space.clone(), Levels::leaf(&owner).cubes(&[BATCH]).build()),
                CubeDim::new_2d(self.plane_width, planes as u32),
            ),
            TestedMicrokernel::CubeColumns { planes, columns } => (
                Partitioning::new(
                    space.clone(),
                    columns_of(&owner, columns, lanes / columns)
                        .planes(&[(REDUCED, planes)])
                        .walk_every(&[REDUCED])
                        .cubes(&[BATCH])
                        .build(),
                ),
                CubeDim::new_2d(self.plane_width, planes as u32),
            ),
            TestedMicrokernel::CubeStrided { planes }
            | TestedMicrokernel::CubePartial { planes } => (
                Partitioning::new(space.clone(), leaf.cubes(&[BATCH]).build()),
                CubeDim::new_2d(self.plane_width, planes as u32),
            ),
            TestedMicrokernel::CubeStaged { planes } => (
                Partitioning::new(
                    space.clone(),
                    self.cube(&hw, planes)
                        .levels(&owner)
                        .cubes(&[BATCH])
                        .build(),
                ),
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
        let edge = Boundary::Identity(self.item.operation.edge_monoid());
        let elements = [self.item.input, self.item.accumulator, self.out];
        let input = bound(input, self.item.input, self.width, &input_dimensions, edge);
        let out = bound(out, self.out, out_width, &out_dimensions, Boundary::Zero);
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            match self.microkernel {
                TestedMicrokernel::UnitDirect => unit_reduce_kernel::launch_unchecked::<F>(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    cells,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    UnitDirectReduceMicrokernel::new(self.item, LANE_SLOTS),
                    elements,
                ),
                TestedMicrokernel::UnitCarrierWrite => unit_carrier_write_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    self.reduced,
                    UnitCarrierWriteMicrokernel::new(self.item, self.row_lanes()),
                    [self.item.accumulator, self.out],
                ),
                TestedMicrokernel::UnitStridedReduce { stride } => {
                    unit_strided_reduce_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        UnitStridedReduceMicrokernel::new(self.item, stride),
                        elements,
                    )
                }
                TestedMicrokernel::UnitWalkReduce { .. } => {
                    unit_walk_reduce_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        self.reduced,
                        UnitWalkReduceMicrokernel::new(self.item, self.microkernel.stride_on(&hw)),
                        elements,
                    )
                }
                TestedMicrokernel::UnitWalkWrite { .. } => {
                    unit_walk_write_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        self.microkernel.values_of(self.reduced),
                        UnitWalkWriteMicrokernel::new(self.item),
                        [self.item.accumulator, self.out],
                    )
                }
                TestedMicrokernel::UnitPartialStore => unit_partial_store_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    UnitPartialStoreMicrokernel::new(self.item),
                    self.item.accumulator,
                ),
                TestedMicrokernel::UnitPartialsReduce => {
                    unit_partials_reduce_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        self.reduced,
                        UnitPartialsReduceMicrokernel::new(self.item, self.row_lanes()),
                        [self.item.accumulator, self.out],
                    )
                }
                TestedMicrokernel::CubePartial { planes } => {
                    cube_partial_reduce_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        CubePartialReduceMicrokernel::new(CubeStagedReduceMicrokernel::new(
                            self.item, lanes, planes, LANE_SLOTS,
                        )),
                        [self.item.input, self.item.accumulator],
                    )
                }
                TestedMicrokernel::PlaneLine => plane_line_reduce_kernel::launch_unchecked::<F>(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    cells,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    PlaneLineReduceMicrokernel::new(self.item, lanes),
                    elements,
                ),
                TestedMicrokernel::PlaneColumns { columns } => {
                    plane_columns_reduce_kernel::launch_unchecked::<F>(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        cells,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        PlaneColumnsReduceMicrokernel::new(self.item, lanes, columns),
                        elements,
                    )
                }
                TestedMicrokernel::CubeLine { planes } => {
                    cube_line_reduce_kernel::launch_unchecked::<F>(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        cells,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        CubeLineReduceMicrokernel::new(self.item, lanes, planes),
                        elements,
                    )
                }
                TestedMicrokernel::CubeColumns { planes, columns } => {
                    cube_columns_reduce_kernel::launch_unchecked::<F>(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        cells,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        CubeColumnsReduceMicrokernel::new(self.item, lanes, planes, columns),
                        elements,
                    )
                }
                TestedMicrokernel::CubeStrided { planes } => {
                    cube_strided_reduce_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        CubeStridedReduceMicrokernel::new(self.item, lanes, planes),
                        elements,
                    )
                }
                TestedMicrokernel::UnitLinesWrite => unit_lines_write_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    self.reduced,
                    UnitLinesWriteMicrokernel::new(self.item),
                    [self.item.accumulator, self.out],
                ),
                TestedMicrokernel::PlaneShuffle => plane_reduce_kernel::launch_unchecked::<F>(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    cells,
                    input,
                    out,
                    launcher.partitioning_arg(),
                    plane,
                    elements,
                ),
                TestedMicrokernel::CubeStaged { planes } => {
                    cube_reduce_kernel::launch_unchecked::<F>(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        self.width,
                        cells,
                        input,
                        out,
                        launcher.partitioning_arg(),
                        self.cube(&hw, planes),
                        elements,
                    )
                }
                TestedMicrokernel::HypercubeLastArrival { .. } => {
                    unreachable!("a hypercube splits its rows apart")
                }
            }
        }
        Ok(())
    }

    /// The lanes of a line that hold one row: all of them along the reduced dimension, one across
    /// it, where each lane is its own row's.
    fn row_lanes(&self) -> usize {
        match LineDirection::of(self.item.lines_along) {
            LineDirection::Along => self.width,
            LineDirection::Across => 1,
        }
    }

    /// The cube microkernel of `planes` planes this case launches on `hw`.
    fn cube(&self, hw: &Hardware, planes: usize) -> CubeStagedReduceMicrokernel {
        CubeStagedReduceMicrokernel::new(self.item, self.plane_width as usize, planes, LANE_SLOTS)
            .across_rows(hw, self.width, self.rows)
    }

    /// Every row split into `parts` equal parts, a cube of `planes` planes a part: the input laid
    /// `input[o, p, r]` along the reduced dimension, `input[p, r, o]` across it. The partials and
    /// the cubes' counters are the launch's own.
    fn launch_split<F: ReduceCarrierFamily>(
        &self,
        client: &Client,
        input: TensorArg,
        out: TensorArg,
        planes: usize,
        parts: usize,
    ) -> Result<(), Decline> {
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
        ) = match LineDirection::of(self.item.lines_along) {
            LineDirection::Along => (
                1,
                [BATCH, PARTS, REDUCED],
                [BATCH, SLOTS],
                [BATCH, PARTIAL_LINES],
                parts * cells * self.width,
                part / self.width,
            ),
            LineDirection::Across => (
                self.width,
                [PARTS, REDUCED, BATCH],
                [SLOTS, BATCH],
                [PARTIAL_LINES, BATCH],
                parts * cells,
                part,
            ),
        };
        let space = Space::new(&[
            (BATCH, self.rows),
            (REDUCED, part),
            (SLOTS, cells),
            (PARTS, parts),
            (PARTIAL_LINES, partial_lines),
        ]);
        let leaf = Levels::leaf(&[
            (BATCH, owned),
            (REDUCED, part),
            (SLOTS, cells),
            (PARTS, 1),
            (PARTIAL_LINES, partial_lines),
        ]);
        let partitioning = Partitioning::new(space.clone(), leaf.cubes(&[BATCH, PARTS]).build());
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: CubeDim::new_2d(self.plane_width, planes as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let partial_shape = match LineDirection::of(self.item.lines_along) {
            LineDirection::Along => [self.rows, partial_lines],
            LineDirection::Across => [partial_lines, self.rows],
        };
        let index = ElemType::UInt(UIntKind::U32);
        let partials = TensorHandle::zeros(client, partial_shape, self.item.accumulator);
        let arrivals = TensorHandle::zeros(client, [self.rows], index);
        // Past its edge the input reads the identity of the operation's edge monoid; the outputs and
        // the partials are written, never read past theirs.
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
                hypercube_indexed_reduce_kernel::launch_unchecked::<F>(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    cells,
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
                hypercube_reduce_kernel::launch_unchecked::<F>(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    self.width,
                    cells,
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
fn unit_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);

        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &row,
                &mut row_out,
                &ComptimeOption::new_None(),
            );
    }
}

/// Every unit's one line of `input`, taken as its carrier of values alone and written to `out`
/// by the write microkernel, a mean dividing by `values`.
///
/// Its operands' leaves:
/// - `input`: one line a unit, the values of a row along the reduced dimension, or a value of a
///   row a lane across it.
/// - `out`: one tensor of a cell a row.
#[cube(launch_unchecked)]
fn unit_carrier_write_kernel<Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitCarrierWriteMicrokernel,
    #[define(Acc, Out)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);
        let carrier = row.flat::<Acc, N>().read(0);

        microkernel
            .runtime()
            .write::<Out, Out, Acc, N, Vector<Acc, N>>(
                carrier,
                &mut row_out,
                &ComptimeOption::new_None(),
                values,
            );
    }
}

/// Every unit's row of `input` read by the strided read microkernel into a carrier of values
/// alone, its lines from the first on, every `stride`-th, the carrier's line written to `out` as
/// it is.
///
/// Its operands' leaves:
/// - `input`: the rows, one value a line along the reduced dimension, or a row a lane across it.
/// - `out`: one tensor of a cell a row.
#[cube(launch_unchecked)]
fn unit_strided_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitStridedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);
        let count = row.flat::<In, N>().shape();

        let carrier = microkernel
            .runtime()
            .reduce::<In, Acc, N, Vector<Acc, N>>(&row, 0, count, 0);
        let mut lines = row_out.flat_mut::<Out, N>();
        lines.write(0, Vector::cast_from(carrier));
    }
}

/// Every plane's rows, its units lying across lines of them, read by the walked read microkernel
/// into a carrier of values alone, each column's first unit's line written to `out` as it is.
///
/// Its operands' leaves:
/// - `input`: the rows, a row a lane, across the reduced dimension.
/// - `out`: one tensor of a cell a row.
#[cube(launch_unchecked)]
fn unit_walk_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitWalkReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for plane in &partitioning {
        let carrier = microkernel
            .runtime()
            .reduce::<In, Acc, N, Vector<Acc, N>>(&plane, &x, values);
        for unit in plane.walk().region(0).leaves() {
            let mut cell_out = o.at(&unit);
            if unit.origin(REDUCED) == 0 {
                cell_out
                    .flat_mut::<Out, N>()
                    .write(0, Vector::cast_from(carrier));
            }
        }
    }
}

/// Every plane's units' lines of the walk's first step, each taken as its carrier of values alone,
/// and each column's first unit's written to `out` by the walked write microkernel, a mean dividing
/// by `values`.
///
/// Its operands' leaves:
/// - `input`: the rows, a row a lane, across the reduced dimension.
/// - `out`: one tensor of a cell a row.
#[cube(launch_unchecked)]
fn unit_walk_write_kernel<Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitWalkWriteMicrokernel,
    #[define(Acc, Out)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for plane in &partitioning {
        let mut carrier = Vector::<Acc, N>::new(Acc::from_int(0));
        for unit in plane.walk().region(0).leaves() {
            let cell = x.at(&unit);
            carrier = cell.flat::<Acc, N>().read(0);
        }
        microkernel
            .runtime()
            .write::<Out, Out, Acc, N, Vector<Acc, N>>(
                &plane,
                carrier,
                &o,
                &ComptimeOption::new_None(),
                values,
            );
    }
}

/// Every unit's one line of `input`, taken as its carrier of values alone and stored to `out` as
/// its one partial.
///
/// Its operands' leaves:
/// - `input`: one line a unit.
/// - `out`: the partials, one line a unit.
#[cube(launch_unchecked)]
fn unit_partial_store_kernel<Acc: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitPartialStoreMicrokernel,
    #[define(Acc)] _element: ElemType,
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut partials = o.at(&unit);
        let row = x.at(&unit);

        microkernel.runtime().store::<Acc, N, Vector<Acc, N>>(
            row.flat::<Acc, N>().read(0),
            &mut partials,
            &ComptimeOption::new_None(),
            0,
        );
    }
}

/// Every unit's row of `input`, each line taken as a partial of values alone, reduced by the
/// partials' microkernel, a mean dividing by `values`.
///
/// Its operands' leaves:
/// - `input`: the partials of each row, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row.
#[cube(launch_unchecked)]
fn unit_partials_reduce_kernel<Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitPartialsReduceMicrokernel,
    #[define(Acc, Out)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let partials = x.at(&unit);
        let parts = partials.flat::<Acc, N>().shape();

        microkernel
            .runtime()
            .reduce::<Out, Out, Acc, N, Vector<Acc, N>>(
                &partials,
                &ComptimeOption::new_None(),
                parts,
                &mut row_out,
                &ComptimeOption::new_None(),
                values,
            );
    }
}

/// Every cube's row, reduced by the cube into a carrier of values alone and stored to `out` as its
/// one partial.
///
/// Its operands' leaves:
/// - `input`: the rows, one value a line along the reduced dimension, or a row a lane across it.
/// - `out`: the partials, one line a row.
#[cube(launch_unchecked)]
fn cube_partial_reduce_kernel<In: Numeric, Acc: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubePartialReduceMicrokernel,
    #[define(In, Acc)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        let mut partials = o.at(&cube);
        let row = x.at(&cube);

        microkernel.runtime().reduce::<In, Acc, N, Vector<Acc, N>>(
            &row,
            &mut partials,
            &ComptimeOption::new_None(),
            0,
            0,
        );
    }
}

/// Every plane's row, reduced by the plane one line of rows wide.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn plane_line_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneLineReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for plane in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &plane,
                &x,
                &o,
                &ComptimeOption::new_None(),
            );
    }
}

/// Every plane's rows, reduced by the plane several lines of rows wide.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn plane_columns_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneColumnsReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for plane in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &plane,
                &x,
                &o,
                &ComptimeOption::new_None(),
            );
    }
}

/// Every cube's row, reduced by the cube one line of rows wide.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn cube_line_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeLineReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &cube,
                &x,
                &o,
                &ComptimeOption::new_None(),
            );
    }
}

/// Every cube's rows, reduced by the cube several lines of rows wide.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn cube_columns_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeColumnsReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &cube,
                &x,
                &o,
                &ComptimeOption::new_None(),
            );
    }
}

/// Every cube's row, read by the cube's strided microkernel into a carrier of values alone, its
/// first unit's line written to `out` as it is.
///
/// Its operands' leaves:
/// - `input`: the rows, one value a line along the reduced dimension, or a row a lane across it.
/// - `out`: one tensor of a cell a row.
#[cube(launch_unchecked)]
fn cube_strided_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeStridedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        let row = x.at(&cube);
        let count = row.flat::<In, N>().shape();

        let carrier = microkernel
            .runtime()
            .reduce::<In, Acc, N, Vector<Acc, N>>(&row, 0, count);
        let mut lines = row_out.flat_mut::<Out, N>();
        if UNIT_POS == 0 {
            lines.write(0, Vector::cast_from(carrier));
        }
    }
}

/// Every unit's one line of `input`, taken as its carrier of values alone and written to `out` by
/// the lines' write microkernel, as wide as it is, a mean dividing by `values`.
///
/// Its operands' leaves:
/// - `input`: one line a unit.
/// - `out`: one line a unit.
#[cube(launch_unchecked)]
fn unit_lines_write_kernel<Acc: Numeric, Out: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] values: usize,
    #[comptime] microkernel: UnitLinesWriteMicrokernel,
    #[define(Acc, Out)] _elements: [ElemType; 2],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for unit in &partitioning {
        let mut row_out = o.at(&unit);
        let row = x.at(&unit);

        microkernel
            .runtime()
            .write::<Out, Out, Acc, N, N, Vector<Acc, N>>(
                row.flat::<Acc, N>().read(0),
                &mut row_out,
                &ComptimeOption::new_None(),
                values,
            );
    }
}

/// Every plane's row, reduced by the plane.
///
/// Its operands' leaves:
/// - `input`: the rows, laid along or across the reduced dimension.
/// - `out`: one tensor of a cell a row (`k` for a top-k).
#[cube(launch_unchecked)]
fn plane_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for plane in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &plane,
                &x,
                &o,
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
fn cube_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                &cube,
                &x,
                &o,
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

    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        let row = x.at(&cube);
        let mut slots = p.at(&cube);

        let counter = cube.origin(BATCH);
        let part = cube.origin(PARTS);
        let first_line = part * part_lines;

        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
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
fn hypercube_indexed_reduce_kernel<
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
    partial_indices: &mut TileTensor<'_>,
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
    let pi = partial_indices.tile_mut(&partitioning);

    for cube in &partitioning {
        let mut row_out = o.at(&cube);
        let row = x.at(&cube);
        let mut slots = p.at(&cube);

        let counter = cube.origin(BATCH);
        let part = cube.origin(PARTS);
        let first_line = part * part_lines;

        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
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

/// Boxes of rows `owner` side by side, `columns` of them, `per_column` units of a plane in each taking
/// the reduced dimension in turns.
fn columns_of(owner: &[(Dimension, usize)], columns: usize, per_column: usize) -> Levels {
    let mut step = owner.to_vec();
    step.push((REDUCED, 1));
    Levels::leaf(&step).units(&[(BATCH, columns), (REDUCED, per_column)])
}

/// `arg` launched in `element`, `width` wide over `dimensions`, reading `boundary` past its edge.
fn bound(
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
