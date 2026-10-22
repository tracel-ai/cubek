use crate::matmul::dense::cube::CubeMeet;
use crate::matmul::dense::plane::{PlaneShuffleMicrokernel, PlaneShuffleMicrokernelSettings};
use crate::matmul::{
    MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulOrder, MatmulPartitioning,
    MatmulRowsKind, MatmulStep, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::QuantValue;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubecl::tile::{Axis, Levels, Space};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, N, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Every row of one stored tile group of a packed weight, owned by one cube: the columns one
/// stored tile holds, read as one stream.
///
/// A load stores an NVFP4 weight in tiles of the step its device contracts in
/// ([`MatmulStep::stored_on`]), each tile a run of `K` for a few columns, its columns side by side
/// and the tiles of a group one after the other along `K`. Each plane covers the group's columns
/// with its units, a few units a column taking `K`'s scale words in turns, so one step of the plane
/// reads whole tiles, every byte of them. The cube's planes take those steps in turns, so between
/// them they read the group front to back. Each unit sums its share in registers, the units of a
/// column sum across the plane, and the planes add their partials into one accumulator in shared
/// memory, as the device lets them meet ([`CubeMeet`]).
///
/// What a weight stored in tiles wants at a few rows: a plane that owns one column reads a few
/// bytes of each tile and steps over the rest, which its neighbours read later.
///
/// The work below the cube is the plane shuffle's: its units and their reads
/// ([`PlaneShuffleMicrokernel`]), laid across the group's columns rather than down one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeTileGroupMicrokernel {
    /// What each plane runs on its steps: the plane shuffle, its units over the group's columns.
    pub plane: PlaneShuffleMicrokernel,
    /// Planes sharing the group, taking its steps along `K` in turns.
    pub planes_along_k: NonZeroUsize,
    /// How its planes add their partial sums into one.
    pub meet: CubeMeet,
}

/// How many planes share a tile group: a target the group's steps along `K` cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeTileGroupMicrokernelSettings {
    pub planes_along_k: NonZeroUsize,
}

impl CubeTileGroupMicrokernelSettings {
    /// Eight planes: a cube of 256 units on a 32-wide plane, enough loads in flight to keep a long
    /// `K` streaming.
    pub const DEFAULT: Self = Self::new(8);

    /// `planes_along_k` planes sharing a group.
    ///
    /// # Panics
    ///
    /// On no planes.
    pub const fn new(planes_along_k: usize) -> Self {
        Self {
            planes_along_k: NonZeroUsize::new(planes_along_k).unwrap(),
        }
    }
}

impl core::fmt::Display for CubeTileGroupMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes_along_k={}", self.planes_along_k)
    }
}

impl CubeTileGroupMicrokernel {
    /// What a weight is declined with where the device stores it in no tile: a group is the
    /// columns one stored tile holds.
    pub const NO_STORED_TILE: Refusal = Refusal::Declined {
        why: "the tile group cube reads a weight of `e2m1` values the device stores in tiles",
    };

    /// The fewest planes a cube holds: a group one plane owns reads it alone.
    const FEWEST_PLANES: usize = 2;

    /// The plane its planes run before their units are laid over a group: one step of `N` a
    /// unit, every row, the whole plane on `K`.
    const PLANE: PlaneShuffleMicrokernelSettings =
        PlaneShuffleMicrokernelSettings::stepping(1, None);

    /// The levels inside one cube's group, which a routine states the cubes around: its planes'
    /// steps through `K`, dealt to the planes in turns along [`split`](Self::split).
    pub fn tiling(&self) -> Levels {
        let split = self.split();
        self.plane
            .step_levels()
            .expect("candidates hold a plane whose units take K in turns")
            .planes_across(split, self.planes_along_k.get())
            .interleaved(split)
    }

    /// Values of `K` one step of a plane covers, `leaf` its unit's box: a unit's turn by the
    /// plane's units along `K`.
    pub fn step_depth(&self, leaf: &Space) -> usize {
        let (blocks, inside) = blocked(K);
        leaf.extent(blocks) * leaf.extent(inside) * self.plane.units_along_k
    }

    /// The axis its planes split `K` along: the weight's scale blocks.
    pub fn split(&self) -> Axis {
        blocked(K).0
    }

    /// Bytes of shared memory one cube holds over its group: the accumulator, one sum a cell of
    /// the cube's box of the output for each copy its planes meet in ([`CubeMeet::copies`]),
    /// which the tile DSL sizes itself.
    pub fn shared_memory(&self, item: &MatmulWorkItem, partitioning: &MatmulPartitioning) -> usize {
        let cells = (partitioning.rows() * partitioning.columns()) as usize;
        cells * self.meet.copies(self.planes_along_k.get()) * item.sum.size()
    }

    /// The columns of one stored tile of `item`'s weight on `hw`: the step a load stores its
    /// `e2m1` values in. `None` for a weight of other values, or a device that stores none in
    /// tiles.
    fn group_columns(hw: &Hardware, item: &MatmulWorkItem) -> Option<usize> {
        let packing = item.blocks?.packings.rhs?;
        let QuantValue::E2M1 = packing.scheme.value else {
            return None;
        };
        let values = ElemType::Float(FloatKind::E2M1);
        let step = MatmulStep::stored_on(hw, &packing.operand, values, item.elements.lhs)?;
        Some(step.along(N))
    }

    /// The plane shuffle's plane at one column a unit, its units laid over `columns` columns, one
    /// group of them a column, the rest along `K`: `None` where the plane does not divide into
    /// that many groups, `n` into whole groups, or the units of a group the scales' words of `K`.
    fn over_columns(
        plane: PlaneShuffleMicrokernel,
        columns: usize,
    ) -> Option<PlaneShuffleMicrokernel> {
        let blocks = plane.blocks?;
        let space = &blocks.space;
        let width = plane.plane_width();
        let shared = width / plane.units_along_m;
        let units_along_k = shared / columns;
        let laid = PlaneShuffleMicrokernel {
            units_along_k,
            groups_along_n: columns,
            ..plane
        };
        let spreads = space
            .operand(MatmulOperand::Rhs)
            .spread_across(K, space.k, units_along_k)
            .is_ok();
        let whole = shared.is_multiple_of(columns)
            && units_along_k > 0
            && laid.columns() == 1
            && space.n.is_multiple_of(columns);
        (whole && spreads).then_some(laid)
    }
}

impl Microkernel for CubeTileGroupMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = CubeTileGroupMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = MatmulPartitioning;

    const DECLINE: &'static str = "the tile group cube needs a plane width the device commits to, a weight of `e2m1` values it stores in tiles whose scales group K, and two turns of K or more for its planes";

    /// Planes a group, its one knob: eight, then four and sixteen.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [8, 4, 16].map(Self::Settings::new).into()
    }

    /// Its planes' items, a packed weight's.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        if item.blocks.and_then(|blocks| blocks.packings.rhs).is_none() {
            return Err(Self::NO_STORED_TILE);
        }
        PlaneShuffleMicrokernel::admits(item)
    }

    /// The plane shuffle at one step of `N` a unit, its units laid over one stored tile's columns,
    /// with the settings' planes capped at the turns those units take through `K`; its planes
    /// meeting as `hw`'s cubes do ([`CubeMeet::of`]).
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let columns = Self::group_columns(hw, item);
        let meet = CubeMeet::of(hw);
        PlaneShuffleMicrokernel::candidates(hw, item, &Self::PLANE)
            .filter_map(move |plane| Self::over_columns(plane, columns?))
            .filter_map(move |plane| {
                let planes = settings.planes_along_k.get().min(plane.turns()?);
                NonZeroUsize::new(planes)
                    .filter(|planes| planes.get() >= Self::FEWEST_PLANES)
                    .map(|planes_along_k| Self {
                        plane,
                        planes_along_k,
                        meet,
                    })
            })
    }

    /// Nothing: against a packed weight every width follows from its packing.
    fn vectorize(&self, _: &MatmulPartitioning, _: &mut Vectorization<MatmulOperand>) {}

    /// Its planes' work: the plane shuffle's.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        _: &MatmulPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.plane.estimate(item, &(), planes, estimate);
    }
}

/// Its plane microkernel, and how many share a group.
impl core::fmt::Display for CubeTileGroupMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} columns={} planes_along_k={}",
            self.plane, self.plane.groups_along_n, self.planes_along_k
        )
    }
}

impl MatmulMicrokernel for CubeTileGroupMicrokernel {
    /// Its planes hold every row.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    /// The packed weight is bound over the axes the product names, its words where they lie.
    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        MatmulOperands::all(MatmulOrder::Stated)
    }
}
