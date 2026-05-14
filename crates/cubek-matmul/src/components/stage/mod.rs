// ! Performs tiled matrix multiplication using shared memory.
// ! Manages unit/plane coordination

#![allow(clippy::type_complexity)]

pub mod partition_matmul;
pub mod partitioner;
pub mod stage_matmul;
mod stage_memory;

pub use cubek_std::tile::{PlanePartitioner, UnitPartitioner};
pub use partition_matmul::*;
pub use partitioner::{StagePartitioner, partition_coordinates};
pub use stage_matmul::*;
pub use stage_memory::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Number of stages in one shared memory, i.e. buffers for double buffering
pub struct NumStages {
    pub lhs: u32,
    pub rhs: u32,
}

impl From<(u32, u32)> for NumStages {
    fn from(value: (u32, u32)) -> Self {
        NumStages {
            lhs: value.0,
            rhs: value.1,
        }
    }
}
