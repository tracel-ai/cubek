use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::{LocalTile, LocalTileLayout, LocalTileLayoutExpand};

use crate::components::tile::softmax::{FragmentMask, SoftmaxLayout, SoftmaxLayoutExpand};

#[cube]
impl SoftmaxLayout for LocalTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        LocalTileLayout::absolute_pos(self, local_pos)
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        LocalTileLayout::num_units_per_row(self)
    }
}

#[cube]
impl<E: Numeric> FragmentMask for LocalTile<E> {
    type Layout = LocalTileLayout;
}
