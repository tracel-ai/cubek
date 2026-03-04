use cubecl::prelude::*;

use crate::tile::strided_tile::StridedTile;

/// Kind (family) of the tiles returned by a stage and ingested by a tile matmul reader
pub trait TileKind<IO: SliceVisibility = ReadOnly>: CubeType + Send + Sync + 'static {
    /// Concrete tile instantiated with the element type
    type Tile<E: Numeric>: CubeType;
}

/// Tile is a slice of memory with a stride
#[derive(CubeType)]
pub struct Strided {}

/// Tile is a single value that gets filled in everywhere
#[derive(CubeType)]
pub struct Filled {}

impl<IO: SliceVisibility> TileKind<IO> for Strided {
    type Tile<E: Numeric> = StridedTile<E, IO>;
}

impl TileKind<ReadOnly> for Filled {
    type Tile<E: Numeric> = E;
}

impl<Inner: TileKind<IO>, IO: SliceVisibility> TileKind<IO> for Option<Inner> {
    type Tile<E: Numeric> = Option<Inner::Tile<E>>;
}

pub type Tile<K, E> = <K as TileKind>::Tile<E>;
pub type TileMut<K, E> = <K as TileKind<ReadWrite>>::Tile<E>;
