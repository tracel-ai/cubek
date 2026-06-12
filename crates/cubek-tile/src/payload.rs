//! The [`Tile`]'s backing store.

use cubecl::prelude::*;

use super::*;

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand.
#[derive(CubeType)]
pub enum Payload<T: CubePrimitive> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// MMA-unit-resident, not addressable (no memory view); contraction is `cmma::execute`.
    Cmma(CmmaData<T>),
}
