//! What a [`Memory`] is: its [`Backing`], its [`Store`], and how it may be touched ([`Access`]).

use cubecl::{
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
};

use crate::*;

/// A lifetime-erased buffer, its fixed `layout`, and the window this tile looks at.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Memory<T: Numeric> {
    /// Which memory the bytes sit in.
    #[cube(comptime)]
    pub(crate) address: AddressSpace,
    /// What the bytes are and mean.
    pub(crate) store: Store<T>,
    /// How a logical coordinate becomes a buffer offset; fixed at construction.
    pub(crate) layout: BufferLayout,
    /// The region of the physical buffer this tile covers; narrowed by [`at`](Tile::at).
    pub(crate) window: Window,
    /// How the tile's logical axes address the buffer's physical ones; fixed at construction.
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// The runtime half of [`projection`](Self::projection): its coefficients and window phase.
    pub(crate) map: RuntimeMap,
    /// The runtime constant terms of the projection, one signed value per dynamic-offset axis.
    pub(crate) offsets: Coords<i32>,
    /// The window origin's line offset, accumulated across [`at`](Tile::at)s.
    pub(crate) window_start: u32,
    /// How this store may be touched; decided at construction.
    #[cube(comptime)]
    pub(crate) access: Access,
    /// What the plane's units are to these cells, stamped across [`at`](Tile::at)s.
    #[cube(comptime)]
    pub(crate) unit_share: UnitShare,
    /// What one instance holds of these cells; read by accumulators only.
    #[cube(comptime)]
    pub(crate) split_share: SplitShare,
    /// What the accumulation being lowered starts from ([`InitFrom`]).
    #[cube(comptime)]
    pub(crate) init_from: InitFrom,
    /// Where this tile's cells sit in the buffer they were filled from; `Some` only for a gathered
    /// stage.
    pub(crate) source_window: ComptimeOption<SourceWindow>,
    /// Whether this operand lands on its way to a tensor-core fragment.
    #[cube(comptime)]
    pub(crate) lands: bool,
    /// The scales these values carry ([`Tile::mul`](crate::Tile::mul)); empty when none.
    pub(crate) factor: Factor,
    /// The table these values index ([`Tile::lookup`](crate::Tile::lookup)); empty when none.
    pub(crate) codebook: Codebook,
}

/// Which memory a [`Memory`] tile's buffer sits in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AddressSpace {
    Global,
    Shared,
}

/// What backs a [`Memory`]'s values: an addressable buffer or an erased call.
/// Address-shaped operations on a call are a comptime panic.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
#[allow(dead_code)] // Built through the derived `new_*` expand constructors.
pub(crate) enum Backing<T: Numeric> {
    /// Bytes this kernel addresses directly, grouped into lines at `vector_size`.
    Buffer(Box<[T]>),
    /// A write-only destination that is not memory ([`ErasedTensor`]).
    WriteCall(ErasedTensor<T, WriteOnly>),
    /// A read-only producer that is not memory.
    ReadCall(ErasedTensor<T, ReadOnly>),
}

/// What a [`Memory`]'s values are: their backing, line width and packing.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Store<T: Numeric> {
    /// What backs the values.
    pub(crate) backing: Backing<T>,
    /// Physical line size of the destination, `1` when unvectorized.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// How the buffer's values sit in it, from [`TileSpec::packed`].
    #[cube(comptime)]
    pub(crate) packing: Packing,
}

#[cube]
impl<T: Numeric> Store<T> {
    /// The bytes, for a destination that has an address.
    // `Box<[T]>` is cubecl's owned-slice handle, not a Rust box; `&[T]` is a different kernel type.
    #[allow(clippy::borrowed_box)]
    pub(crate) fn buffer(&self) -> &Box<[T]> {
        match &self.backing {
            Backing::Buffer(buffer) => buffer,
            Backing::WriteCall(_) => panic!(
                "Store::buffer: this tile's backing is written through a call, which has no \
                 address, it can only be written through its layout"
            ),
            Backing::ReadCall(_) => panic!(
                "Store::buffer: this tile's backing is read through a call, which has no \
                 address, it can only be read through its layout (Memory::read_view), so the \
                 slice-shaped paths (a dense run, a re-typed quant storage, a tensor-map load) \
                 are closed to it"
            ),
        }
    }

    /// Whether the values have an address: a buffer, rather than a call.
    pub(crate) fn has_address(&self) -> comptime_type!(bool) {
        match &self.backing {
            Backing::Buffer(_) => comptime!(true),
            Backing::WriteCall(_) | Backing::ReadCall(_) => comptime!(false),
        }
    }

    /// The mutable twin of [`buffer`](Self::buffer).
    #[allow(clippy::borrowed_box)]
    pub(crate) fn buffer_mut(&mut self) -> &mut Box<[T]> {
        match &mut self.backing {
            Backing::Buffer(buffer) => buffer,
            Backing::WriteCall(_) => panic!(
                "Store::buffer_mut: this tile's backing is written through a call, which has no \
                 address, it can only be written through its layout"
            ),
            Backing::ReadCall(_) => panic!(
                "Store::buffer_mut: this tile's backing is read through a call, which is \
                 read-only"
            ),
        }
    }
}

/// How a [`Memory`] may be touched; plain comptime data.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Access {
    /// Whether the window still covers the whole buffer.
    pub whole: bool,
    pub overhang: Overhang,
    /// What a write here does to the cell it lands on.
    pub write: Write,
    /// The units that share a cooperative fill of this window.
    pub fill: FillUnits,
    /// What the storage tiles are to this window.
    pub storage: Storage,
}

/// The units that share a cooperative fill of a window: every unit of the cube, or the units of
/// one plane, for a stage that plane owns and fills alone ([`Stages::smem`](crate::Stages::smem)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FillUnits {
    /// Whose units they are: the cube's, or one plane's.
    pub scope: ComputeScope,
    /// How many there are, `0` when unknown: a stage filled from this tile emits its fill
    /// straight-line when it knows. The launch's cube size, or one plane's width.
    pub count: usize,
}

impl FillUnits {
    /// Every unit of the cube, `count` of them (`0` when unknown).
    pub(crate) fn cube(count: usize) -> Self {
        FillUnits {
            scope: ComputeScope::Cube,
            count,
        }
    }
}

/// This unit's position among the units `fill` names, which a cooperative fill takes its lines
/// at: its position in the cube, or in its plane.
///
/// A plane is the launch's `x` ([`Partitioning::cube_dim`]), so a unit's position in its plane
/// is `UNIT_POS_X` and the plane's width `CUBE_DIM_X`.
#[cube]
pub(crate) fn fill_worker(#[comptime] fill: FillUnits) -> usize {
    match comptime!(fill.scope) {
        ComputeScope::Cube => UNIT_POS as usize,
        ComputeScope::Plane => UNIT_POS_X as usize,
        ComputeScope::Unit => comptime!(panic!(
            "fill_worker: a cooperative fill is shared by a cube or a plane, never one unit"
        )),
    }
}

/// How many units `fill` names at runtime: the cube's, or one plane's ([`fill_worker`]).
#[cube]
pub(crate) fn fill_workers(#[comptime] fill: FillUnits) -> usize {
    match comptime!(fill.scope) {
        ComputeScope::Cube => CUBE_DIM as usize,
        ComputeScope::Plane => CUBE_DIM_X as usize,
        ComputeScope::Unit => comptime!(panic!(
            "fill_workers: a cooperative fill is shared by a cube or a plane, never one unit"
        )),
    }
}

/// What a write to a store does to the cell it lands on; stated by the binding operand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Write {
    /// Replaces the cell.
    Replace,
    /// Adds into the cell, atomically.
    Accumulate,
}

impl Write {
    /// Refuse an accumulation `split` leaves in pieces unless this write adds them.
    pub(crate) fn admits(self, split: SplitShare, site: &str) {
        match (split, self) {
            (SplitShare::Whole, _) | (SplitShare::Partial, Write::Accumulate) => {}
            (SplitShare::Partial, Write::Replace) => panic!(
                "{site}: this accumulator's cells are split across planes or cubes and its \
                 destination replaces rather than accumulates, so every partial but one would be \
                 lost. \
                 A contracted axis distributed across planes or cubes gives each instance a \
                 slice of the contraction, and none of them holds a whole cell. \
                 Drain into an accumulating destination (bind it as an `AccumulateArg`), \
                 distribute the contraction across the plane's units instead \
                 (`distribute(units(n), ..)`, combined in the plane's registers), or give the \
                 output an axis of its own for the split."
            ),
        }
    }
}

/// How a store relates to the window overhanging its valid data (past [`Window`]'s `bound`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Overhang {
    /// Impossible: the buffer is allocated to exactly the tile (smem).
    Never,
    /// Excluded at launch: every shape divides its tiling (unchecked gmem).
    Fits,
    /// Reads/writes past `bound` are masked per the window's [`Boundary`].
    Masked,
}

/// Boundary handling mode for out-of-bounds reads/writes, carried by `Window`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Boundary {
    /// Out-of-bounds reads return zero; writes are skipped.
    Zero,
    /// Out-of-bounds reads/writes clamp to the edge cell.
    Clamp,
}

impl Overhang {
    /// The flag a [`Masked`] is built with.
    pub fn masks(&self) -> bool {
        matches!(self, Overhang::Masked)
    }
}

/// Whether a read still proves its own bounds, stated by the reader.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Guard {
    /// Mask the overhang and apply the window's [`Boundary`] on every access.
    Checked,
    /// The reader proved its whole box is in bounds; reading past it is out of bounds, not masked.
    Proved,
}

impl Guard {
    /// Whether this guard still costs a test per access.
    pub fn checks(self) -> bool {
        matches!(self, Guard::Checked)
    }
}

/// What a storage tile is to the window an operand is read through.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Storage {
    /// Untiled storage: every window lies inside the one storage tile.
    Strided,
    /// Storage-tiled, and the window may span several tiles.
    /// The level is the one whose tile the storage tile is; `None` if no level's.
    Tiled(Option<usize>),
    /// Storage-tiled and inside one storage tile: one contiguous run from its origin.
    Contiguous,
}
