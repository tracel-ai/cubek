use cubecl::prelude::CubeType;

use crate::components::{
    global::read::{FullLoadingStrategy, PartialLoadingStrategy},
    stage::StageFamily,
};

#[derive(Copy, Clone, CubeType)]
/// Identifier for the stage in global double buffering
pub enum StageBuffer {
    /// First buffer
    A,
    /// Second buffer
    B,
}

impl StageBuffer {
    pub fn to_index(&self) -> u32 {
        match self {
            StageBuffer::A => 0,
            StageBuffer::B => 1,
        }
    }
}

#[derive(CubeType, Clone)]
/// Comptime counter for loading tasks
pub struct TaskCounter {
    #[cube(comptime)]
    pub counter: u32,
}

pub type FullLoaderStage<RC, L, IP> = <<L as FullLoadingStrategy<RC>>::Stage as StageFamily>::Stage<
    IP,
    <L as FullLoadingStrategy<RC>>::TilingLayout,
>;
pub type PartialLoaderStage<RC, L, IP> =
    <<L as PartialLoadingStrategy<RC>>::Stage as StageFamily>::Stage<
        IP,
        <L as PartialLoadingStrategy<RC>>::TilingLayout,
    >;
