//! The [`Tile`]'s backing store.

use cubecl::prelude::*;

use super::*;

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand.
#[derive(CubeType)]
pub enum TileKind<T: Numeric> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// MMA-unit-resident, not addressable (no memory view); contraction is `cmma::execute`.
    Cmma(CmmaData<T>),
    /// A TMA tensor-map source: not element-addressable, can only be the source of a
    /// [`stage_from`](Tile::stage_from) into shared memory, which lowers to a hardware bulk copy.
    /// Built but dormant — no launch-side constructor wires it yet (see [`Tile::from_tensor_map`]).
    TmaGmem(TmaData<T>),
}
