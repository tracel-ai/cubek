//! What a tile holds ([`TileKind`]): the one enum every verb on a tile dispatches on.

use cubecl::prelude::*;

use crate::*;

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand. `Clone` copies the handle, not the cells: sound only where nothing rewrites the buffer.
// A kernel value: every variant is what the kernel holds, and a boxed variant is not a cube
// type, so the size difference stays.
#[allow(clippy::large_enum_variant)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum TileKind<T: Numeric> {
    /// An addressable buffer with a window, in global or shared memory
    /// ([`address`](Memory::address)).
    Memory(Memory<T>),
    /// One plane-level tile: owned by a plane and sliced across its units, so never addressable
    /// (no memory view). The [`Instruction`] picks the encoding; the contraction is its own.
    PlaneTile(PlaneTile<T>),
    /// The grid of plane tiles one plane owns, `m_tiles × n_tiles`, comptime-indexed; only a
    /// static walk's regions (constant coordinates) can select through it.
    PlanePartition(PlanePartition<T>),
    /// A TMA tensor-map source: not element-addressable, its only sink is a hardware bulk
    /// copy into shared memory. Launched via [`TmaTileArg`](crate::TmaTileArg).
    TmaGmem(TmaData<T>),
    /// A read-only source evaluated from logical coordinates with no backing buffer.
    Procedural(Procedural<T>),
    /// A tile the plane holds in its units, one line to a unit, read at coordinates and reached
    /// by shuffle or through the plane's own window ([`Lines`]).
    Lines(Lines<T>),
}
