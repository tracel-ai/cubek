//! What a [`Memory`] is: the erased buffer it addresses ([`Backing`]), what its values mean
//! ([`Store`]), and how it may be touched ([`Access`] and the comptime flags qualifying a
//! read or a write).

use cubecl::{
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
};

use crate::*;

/// A lifetime-erased buffer, how to address it ([`layout`](BufferLayout)), and which part of it this
/// tile is looking at ([`window`](Window)). The layout is fixed at construction, so a staged smem
/// sub-tile keeps addressing its whole buffer after [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Memory<T: Numeric> {
    /// Which memory the bytes sit in: what a barrier orders and where a fill may write straight.
    #[cube(comptime)]
    pub(crate) address: AddressSpace,
    /// What the bytes are and mean.
    pub(crate) store: Store<T>,
    /// How a logical coordinate becomes a buffer offset. Fixed at construction.
    pub(crate) layout: BufferLayout,
    /// The region of the *physical* buffer this tile covers; narrowed by [`at`](Tile::at).
    pub(crate) window: Window,
    /// How the tile's logical axes address the buffer's physical ones:
    /// [`direct`](Projection::direct) for every non-gather operand, an affine map for a gather.
    /// Fixed at construction, like the layout: `at` moves the window, never the mapping.
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// What [`projection`](Self::projection) only knows in the kernel: its runtime coefficients and
    /// the phase its window origin sits at. [`integral`](RuntimeMap::integral) for every operand
    /// but a runtime-strided or fractionally scaled gather.
    pub(crate) map: RuntimeMap,
    /// The runtime half of the projection's constant terms: one signed value per
    /// [`Offset::Dynamic`](crate::Offset) axis, signed since a padding starts before the buffer.
    /// Not in [`map`](Self::map): it only places the top [`window`](Self::window).
    pub(crate) offsets: Coords<i32>,
    /// The window origin's offset through the layout, accumulated across [`at`](Tile::at)s rather
    /// than re-derived: each descent shifts by a *comptime* edge, so [`step_offset`] folds to a
    /// multiply-add, where re-deriving it would divide per [`window_slice`](Memory::window_slice).
    pub(crate) window_start: u32,
    /// How this store may be touched. All comptime, all decided at construction.
    #[cube(comptime)]
    pub(crate) access: Access,
    /// What the plane's units are to these cells, stamped across [`at`](Tile::at)s, since the
    /// level that spreads an axis is only known on the way down.
    #[cube(comptime)]
    pub(crate) unit_share: UnitShare,
    /// What one instance holds of these cells, stamped across [`at`](Tile::at)s like
    /// [`units`](Self::units), since only each level's whole space still has the axis this
    /// operand's projection dropped. Read by accumulators only; meaningless (`Partial`) elsewhere.
    #[cube(comptime)]
    pub(crate) split_share: SplitShare,
    /// What the accumulation being lowered starts from ([`InitFrom`]), a claim about what the
    /// caller asked for, not the bytes: [`Identity`](InitFrom::Identity) where [`Tile::mm`] or
    /// [`Tile::reduce_axis`] proves its leaf visits each cell once, else [`Cell`](InitFrom::Cell).
    #[cube(comptime)]
    pub(crate) init_from: InitFrom,
    /// Where this tile's cells sit inside the buffer they were *filled from*. `None` for a tile
    /// reading its source directly ([`window`](Self::window) is the source one); `Some` only for
    /// a gathered stage, whose fill replaced out-of-bounds samples its own window cannot name.
    pub(crate) source_window: ComptimeOption<SourceWindow>,
    /// Whether this operand lands on its way to a tensor-core fragment: unpacked and scaled by the
    /// plane's units into plane-owned shared memory ([`Tile::landed`](crate::Tile::landed)).
    /// Opened by [`with_landing`](Tile::with_landing); without one the leaf takes it unscaled only.
    #[cube(comptime)]
    pub(crate) lands: bool,
    /// The scales these values carry, attached by [`Tile::mul`](crate::Tile::mul) and read where
    /// the values are read. Empty is an operand carrying none.
    pub(crate) factor: Factor,
}

/// Which memory a [`Memory`] tile's buffer sits in. The payload is the same either way; the
/// difference is three facts about it: a `sync_cube()` orders shared accesses only, a shared
/// buffer is allocated to exactly its tile and so never overhangs, and a shared stage remembers
/// the window it was filled from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AddressSpace {
    Global,
    Shared,
}

/// What backs a [`Memory`]'s values, and what can be done with them there.
///
/// A [`Buffer`](Backing::Buffer) has an address: it can be read back, sliced, re-typed, staged or
/// handed to a tensor-map load. The erased two end the walk in a *call* (a generated epilogue or
/// producer), so every address-shaped operation on them is a comptime panic, not a fallback.
///
/// Each erased backing serves one layout-addressed view: [`write_view`](Memory::write_view) for a
/// [`WriteCall`](Backing::WriteCall) and [`read_view`](Memory::read_view) for a
/// [`ReadCall`](Backing::ReadCall).
///
/// The visibility markers carry the direction. A destination is written and
/// never read, a producer read and never written, and neither can be handed
/// where the other belongs without the type saying so.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
#[allow(dead_code)] // Built through the derived `new_*` expand constructors.
pub(crate) enum Backing<T: Numeric> {
    /// Bytes this kernel addresses directly. Scalar-typed by Rust-side erasure
    /// only: the real binding/alloc element is `Vector<T, vector_size>`, so
    /// re-grouping to lines at that width is a no-op.
    Buffer(Box<[T]>),
    /// A destination that is not memory: written through its layout and never read,
    /// which is what [`WriteOnly`] states. See [`ErasedTensor`].
    WriteCall(ErasedTensor<T, WriteOnly>),
    /// A producer that is not memory: read through its layout and never written,
    /// which is what [`ReadOnly`] states. The fuse-on-read twin of
    /// [`WriteCall`](Backing::WriteCall).
    ReadCall(ErasedTensor<T, ReadOnly>),
}

/// What a [`Memory`]'s values are and mean: where they go, the width they group into lines at,
/// and, for quantized data, how a *stored* value becomes a *served* one. Reads through
/// [`Tile::flat`] dequantize into `T`; every other element view refuses a quantized tile.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Store<T: Numeric> {
    /// What backs the values.
    pub(crate) backing: Backing<T>,
    /// Physical line size (`Vector<T, vector_size>`) of the destination, `1` when
    /// unvectorized; held comptime so `size!` can read it.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Present when the destination holds quantized data (see [`QuantInfo`]).
    pub(crate) quant: ComptimeOption<QuantInfo>,
    /// How the buffer's values sit in it: whether a stored element *is* a served one, and what a
    /// read has to unpack if it is not. Stated at construction, from the operand's spec
    /// ([`TileSpec::packed`]) or from its scheme where it has one, so no reader re-derives it.
    #[cube(comptime)]
    pub(crate) packing: Packing,
}

#[cube]
impl<T: Numeric> Store<T> {
    /// The bytes, for a destination that has an address.
    ///
    /// Every reader goes through here, so an erased backing meets one message
    /// rather than a different confusion per call site.
    // `Box<[T]>` is cubecl's owned-slice handle rather than a Rust box, and `&[T]` is a different
    // kernel type with different operations (the re-typing and re-grouping every reader below
    // does), so the lint's suggestion does not apply.
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

/// How a [`Memory`] may be touched: whether the fill can write straight through, how the store
/// handles overhang, and how a cooperative fill spreads. Plain data held comptime.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Access {
    /// Whether the window still covers the whole buffer (constructors yes, [`at`](Tile::at) no):
    /// such a tile can be written in physical order.
    pub whole: bool,
    pub overhang: Overhang,
    /// What a write here does to the cell it lands on.
    pub write: Write,
    /// The launch's cube size (units per cube), `0` when unknown: a stage filled from this tile
    /// emits its fill straight-line when it knows how many units share it.
    pub units: usize,
    /// What the storage tiles are to this window. [`at`](crate::Tile::at) carries it down and
    /// turns [`Tiled`](Storage::Tiled) into [`Contiguous`](Storage::Contiguous) at the storage
    /// tile's own level; the buffer's layout itself never changes.
    pub storage: Storage,
}

/// What a write to a store does to the cell it lands on.
///
/// `Replace` is every buffer and every plain sink: the cell is its writer's own. `Accumulate` is
/// what lets a contraction be cut at cube scope: instances each holding a slice of one cell all
/// write it and the store adds, so none knows about the others and no second pass is needed.
///
/// Stated by the operand that binds the store ([`AccumulateArg`]), never derived. A backing
/// cannot be asked what its writes mean: an accumulating sink and a fused epilogue are both calls
/// through a layout, and only the caller knows which it built.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Write {
    /// Replaces the cell.
    Replace,
    /// Adds into the cell, atomically.
    Accumulate,
}

impl Write {
    /// Refuse an accumulation `split` leaves in pieces unless this write adds them. Called where
    /// an accumulator is opened and where it is written, the two places a partial can escape.
    ///
    /// A replacing destination is silently wrong: a register drain stores, so the last instance
    /// erases the rest, and one accumulating in place loses the update.
    /// [`Accumulate`](Write::Accumulate) is the case this lets through.
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

/// How a store relates to the window overhanging its valid data (`origin + pos` past
/// [`Window`]'s `bound`); where gmem and smem genuinely differ.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Overhang {
    /// Structurally impossible: the buffer is allocated to exactly the tile (smem).
    Never,
    /// Possible in principle, excluded at launch: every shape divides its tiling (unchecked gmem).
    Fits,
    /// Possible: reads/writes past `bound` are masked, per the window's [`Boundary`] (zero for
    /// reads and skipped for writes under `Zero`, the edge cell under `Clamp`).
    Masked,
}

/// Boundary handling mode for out-of-bounds reads/writes, carried by [`Window`] (the layer that
/// owns `origin`/`bound`/`signed` and so is the one that can turn an out-of-range coordinate into
/// a valid physical one).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Boundary {
    /// Out-of-bounds reads return zero; writes are skipped.
    Zero,
    /// Out-of-bounds reads/writes clamp to the edge cell.
    Clamp,
}
impl CubeDebug for Boundary {}

impl Overhang {
    /// The flag a [`Masked`] is built with; the one place the states collapse to a bool.
    pub fn masks(&self) -> bool {
        matches!(self, Overhang::Masked)
    }
}

/// Whether a read still proves its own bounds, stated by the reader rather than read off the
/// tile. Comptime, so the arm not taken costs nothing.
///
/// A tile records what it *could* need ([`Overhang`], the window's [`Boundary`]); this records
/// what a particular reader has established it needs, which is the weaker claim and the only one
/// a leaf splitting itself across an edge can make.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Guard {
    /// Mask the overhang and apply the window's [`Boundary`] on every access.
    Checked,
    /// The reader has proved the whole box it will touch lands inside the buffer, so the view
    /// carries neither. Reading through it past that box is out of bounds, not masked.
    Proved,
}

impl Guard {
    /// Whether this guard still costs a test per access.
    pub fn checks(self) -> bool {
        matches!(self, Guard::Checked)
    }
}

/// What a storage tile is to the window an operand is read through.
///
/// A storage-tiled tensor's storage tile is the tile of a level of the kernel's nest, as a scale
/// block is an axis of a scaled matmul: the space owns the block size, so a window that descended
/// through that level lies inside one storage tile by construction, not by a divisibility check.
///
/// Settled by the launch, which has the buffer's real extents and the kernel's levels in hand.
/// A comptime fact in the kernel; [`at`](crate::Tile::at) makes [`Tiled`](Storage::Tiled)
/// [`Contiguous`](Storage::Contiguous).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Storage {
    /// Untiled storage: the whole buffer is one storage tile, addressed by its strides, and every
    /// window lies inside it.
    Strided,
    /// Storage-tiled, and this window may span several storage tiles, so only a layout walk
    /// addresses its cells. The level is the one of the kernel's nest whose tile the storage tile
    /// is: descending through it makes the window [`Contiguous`](Self::Contiguous). `None` where
    /// the storage tile is no level's tile, and then no window is ever known to lie inside one.
    Tiled(Option<usize>),
    /// Storage-tiled and inside one storage tile: one contiguous run from its origin, addressed
    /// affinely by the storage tile's own strides, which is what a fragment load and a stage fill
    /// want.
    Contiguous,
}
