use cubecl::prelude::*;

use crate::components::tile::{
    StridedTile, interleaved::config::InterleavedMatmulConfig, register::UnitFragment,
};

/// Writer for the register matmul fragments.
#[derive(CubeType)]
pub struct InterleavedStageWriter {}

#[cube]
impl InterleavedStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &UnitFragment<A>,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
    }
}
