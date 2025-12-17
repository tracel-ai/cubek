use cubecl::prelude::*;

use crate::{components::tile::StridedTile, definition::as_cmma_layout};

/// Writer using the cmma store function.
#[derive(CubeType)]
pub struct CmmaStageWriter {}

#[cube]
impl CmmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric>(
        tile: &mut StridedTile<V, ReadWrite>,
        fragment: &cmma::Matrix<E>,
    ) {
        let layout = as_cmma_layout(tile.layout);
        let (mut slice, stride) = tile.as_unlined_mut();
        cmma::store(&mut slice, fragment, stride, layout);
    }
}
