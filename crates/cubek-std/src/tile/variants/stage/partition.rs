use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::tile::{Tile, TileScope};

/// Tile kind holding a per-partition collection of instruction-level tiles.
/// Replaces the bespoke `Accumulators<MP, Sc>` wrapper used by the
/// partition-matmul flow; placed inside
/// [`TileKind::Partition`](crate::tile::TileKind) so accumulators participate
/// in the same `.mma` / `.copy_from` dispatch as their constituent tiles.
///
/// The element tiles share the partition's [`TileScope`] `Sc` (so a
/// plane-partitioned matmul carries `Sc = Plane`, a unit-partitioned matmul
/// carries `Sc = Unit`). The partition shape lives as a comptime `(rows,
/// cols)` pair; the `mn`-major flattening matches today's `Accumulators`
/// indexing.
#[derive(CubeType)]
pub struct PartitionTile<N: Numeric, Sc: TileScope, IO: SliceVisibility = ReadWrite> {
    pub tiles: Sequence<Tile<N, Sc, IO>>,
    #[cube(comptime)]
    pub rows: u32,
    #[cube(comptime)]
    pub cols: u32,
    #[cube(comptime)]
    pub _phantom: PhantomData<Sc>,
}
