//! What a [`MemData`] is: the erased buffer it addresses ([`Backing`]), what its values mean
//! ([`Store`]), and how it may be touched ([`Access`] and the comptime flags qualifying a
//! read or a write).

use cubecl::{
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
};

use crate::*;

/// A lifetime-erased buffer, how to address it ([`layout`](GmemLayout)), and which part of it this
/// tile is looking at ([`window`](Window)). The layout is fixed at construction, so a staged smem
/// sub-tile keeps addressing its whole buffer after [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemData<T: Numeric> {
    /// What the bytes are and mean.
    pub(crate) store: Store<T>,
    /// How a logical coordinate becomes a buffer offset. Fixed at construction.
    pub(crate) layout: GmemLayout,
    /// The region of the *physical* buffer this tile covers; narrowed by [`at`](Tile::at).
    pub(crate) window: Window,
    /// How the tile's logical axes address the buffer's physical ones:
    /// [`direct`](Projection::direct) for every non-gather operand, an affine map for one gathered
    /// over an abstract dimension. Fixed at construction, like the layout: `at` moves the window,
    /// never the mapping.
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// What [`projection`](Self::projection) only knows in the kernel: its runtime coefficients and
    /// the phase its window origin sits at. [`integral`](RuntimeMap::integral) for every operand
    /// but a runtime-strided or fractionally scaled gather.
    pub(crate) map: RuntimeMap,
    /// The runtime half of the projection's constant terms: one signed value per
    /// [`Offset::Dynamic`](crate::Offset) axis, since a padding places the window before the
    /// buffer's origin. Not part of [`map`](Self::map): an offset only places the top window,
    /// which [`window`](Self::window) then carries.
    pub(crate) offsets: Coords<i32>,
    /// The window origin's offset through the layout, accumulated across [`at`](Tile::at)s rather
    /// than re-derived: each descent shifts by a *comptime* edge, so [`step_offset`] folds
    /// and this stays a multiply-add. Addressing it from the origin instead would decompose a
    /// runtime coordinate, i.e. integer division per [`window_slice`](MemData::window_slice).
    pub(crate) window_start: u32,
    /// How this store may be touched. All comptime, all decided at construction.
    #[cube(comptime)]
    pub(crate) access: Access,
    /// What the plane's lanes are to these cells. The share is stamped across [`at`](Tile::at)s,
    /// since the level that spreads an axis is consumed on the way down; the work is settled at
    /// construction, being a fact about the whole space.
    #[cube(comptime)]
    pub(crate) lanes: Lanes,
    /// What one instance holds of these cells, settled at construction: only the whole space can
    /// tell a split from a cut whose edge is the whole axis. This and [`lanes`](Self::lanes) are
    /// read by accumulators alone, and both are `Partial` on an operand merely orthogonal to a
    /// split, where they mean nothing.
    #[cube(comptime)]
    pub(crate) split_share: SplitShare,
    /// What the accumulation being lowered right now starts from ([`InitFrom`]). Not a claim about
    /// the bytes, only about what the caller asked for: [`Tile::mm`] and [`Tile::reduce_axis`]
    /// state [`Identity`](InitFrom::Identity) over their own lowering, having proven the leaf
    /// visits each cell once; it is [`Cell`](InitFrom::Cell) elsewhere and rides
    /// [`at`](MemData::at) down.
    #[cube(comptime)]
    pub(crate) init_from: InitFrom,
    /// Where this tile's cells sit inside the buffer they were *filled from*, when that is not the
    /// buffer they live in. `None` for every tile reading its source directly, where
    /// [`window`](Self::window) already is the source window. `Some` only for a gathered stage,
    /// whose fill replaced out-of-bounds samples and whose window can no longer say which.
    pub(crate) source_window: ComptimeOption<SourceWindow>,
}

/// What backs a [`MemData`]'s values, and what can be done with them there.
///
/// These are not three spellings of one thing. A [`Buffer`](Backing::Buffer) has
/// an address: it can be read back, sliced, re-typed, staged into shared memory,
/// handed to a tensor-map load. The erased two have none: the walk ends in a
/// *call*, which is what lets a kernel hand its values to a generated epilogue,
/// or take them from a generated producer, instead of moving them through
/// memory. So every address-shaped operation is a comptime panic rather than a
/// fallback, and what each serves is one layout-addressed view:
/// [`write_view`](MemData::write_view) for a [`WriteCall`](Backing::WriteCall),
/// [`read_view`](MemData::read_view) for a [`ReadCall`](Backing::ReadCall).
///
/// The visibility markers carry the direction. A destination is written and
/// never read, a producer read and never written, and neither can be handed
/// where the other belongs without the type saying so.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
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

/// What a [`MemData`]'s values are and mean: where they go, the width they group into lines at,
/// and, when the destination physically holds quantized data, how a *stored* value becomes a
/// *served* one. Reads through [`Tile::flat`] dequantize into `T`; every other element view
/// refuses a quantized tile.
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
    // `Box<[T]>` is cubecl's owned-slice handle rather than a Rust box, and `&[T]`
    // is a different kernel type with a different set of operations (the
    // re-typing and re-grouping every reader below does), so the lint's
    // suggestion does not apply.
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
                 address, it can only be read through its layout (MemData::read_view), so the \
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

/// How a [`MemData`] may be touched: whether the fill can write straight through, how the store
/// handles overhang, and how a cooperative fill spreads. Plain data held comptime, like the
/// [`StagePlan`] it carries.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Access {
    /// Whether the window still covers the whole buffer (constructors yes, [`at`](Tile::at) no):
    /// such a tile can be written in physical order.
    pub whole: bool,
    pub overhang: Overhang,
    /// What a write here does to the cell it lands on.
    pub write: Write,
    /// Where this operand lives at each level below, plus the [`StageStorage`] layout and launch
    /// cube size its materialized levels take. Carried from the operand's [`TileSpec`] so a fill
    /// re-derives none of them.
    pub stage: StagePlan,
}

/// What a write to a store does to the cell it lands on.
///
/// `Replace` is every buffer and every plain sink: the cell is its writer's own, so the value
/// that lands is the value that stays. `Accumulate` is what lets a contraction be cut at cube
/// scope: instances that each hold a slice of one cell all write it, and the store adds rather
/// than overwrites, so none of them has to know about the others and no second pass is needed.
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
    /// Refuse an accumulating write from a drain that cannot elect one writer for it.
    ///
    /// A hardware fragment stores through its own intrinsic, over a slice of the destination or
    /// over its lanes' own positions, and neither leaves anywhere to put the election accumulating
    /// needs. The register block is the one drain that writes cell by cell and so can.
    pub(crate) fn validate_fragment_drain(self, fragment: &str) {
        match self {
            Write::Replace => {}
            Write::Accumulate => panic!(
                "{fragment}: a hardware fragment stores through its own intrinsic and elects no \
                 writer, so it cannot drain into a destination that folds. Contract through \
                 Instruction::Registers, whose block drains cell by cell."
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

impl Overhang {
    /// The flag a [`MaskedView`] is built with; the one place the states collapse to a bool.
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
