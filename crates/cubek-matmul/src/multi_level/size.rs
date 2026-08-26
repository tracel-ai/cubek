use cubecl::prelude::*;
use cubek_std::{define_3d_size_base, impl_3d_size_from_tuple};

// Number of elements in a tile
define_3d_size_base!(TileSize, u32);
impl_3d_size_from_tuple!(TileSize, u32, u8);
impl_3d_size_from_tuple!(TileSize, u32, u32);
impl_3d_size_from_tuple!(TileSize, u32, i32);
impl_3d_size_from_tuple!(TileSize, u32, u16);
impl_3d_size_from_tuple!(TileSize, u32, usize);

// Number of tiles in a stage partition
define_3d_size_base!(PartitionSize, u8);
impl_3d_size_from_tuple!(PartitionSize, u8, u8);
impl_3d_size_from_tuple!(PartitionSize, u8, u32);
impl_3d_size_from_tuple!(PartitionSize, u8, i32);
impl_3d_size_from_tuple!(PartitionSize, u8, u16);
impl_3d_size_from_tuple!(PartitionSize, u8, usize);

// Number of partitions in a stage
define_3d_size_base!(StageSize, u8);
impl_3d_size_from_tuple!(StageSize, u8, u8);
impl_3d_size_from_tuple!(StageSize, u8, u32);
impl_3d_size_from_tuple!(StageSize, u8, i32);
impl_3d_size_from_tuple!(StageSize, u8, u16);
impl_3d_size_from_tuple!(StageSize, u8, usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
/// Number of global matmul blocks computed by a single cube.
pub struct GlobalPartitionSize {
    pub m: u32,
    pub n: u32,
    pub batches: u32,
}

impl GlobalPartitionSize {
    pub fn new(m: u32, n: u32, batches: u32) -> Self {
        Self { m, n, batches }
    }
}
