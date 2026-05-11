use cubecl::prelude::*;

use crate::tile::variants::strided::StridedTile;

/// Kind (family) of the tiles returned by a stage and ingested by a tile matmul
/// reader. Distinct from the [`TileKind`](crate::tile::TileKind) enum that
/// identifies storage variants of a [`Tile`](crate::tile::Tile); this trait
/// describes the static *family* of tiles a stage emits.
pub trait StageTileKind<IO: SliceVisibility = ReadOnly>: CubeType + Send + Sync + 'static {
    /// Concrete tile instantiated with the element type
    type Tile<E: Numeric, N: Size>: CubeType;
}

/// Tile is a slice of memory with a stride
#[derive(CubeType)]
pub struct Strided {}

/// Tile is a single value that gets filled in everywhere
#[derive(CubeType)]
pub struct Filled {}

impl<IO: SliceVisibility> StageTileKind<IO> for Strided {
    type Tile<E: Numeric, N: Size> = StridedTile<E, N, IO>;
}

impl StageTileKind<ReadOnly> for Filled {
    type Tile<E: Numeric, N: Size> = E;
}

impl<Inner: StageTileKind<IO>, IO: SliceVisibility> StageTileKind<IO> for Option<Inner> {
    type Tile<E: Numeric, N: Size> = ComptimeOption<Inner::Tile<E, N>>;
}
