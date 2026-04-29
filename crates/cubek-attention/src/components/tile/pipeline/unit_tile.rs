use cubecl::{prelude::*, std::tensor::layout::Coords2d, {self}};
use cubek_std::tile::{UnitTile, UnitTileLayout, UnitTileLayoutExpand};

use crate::components::tile::softmax::{FragmentMask, SoftmaxLayout, SoftmaxLayoutExpand};

#[cube]
impl SoftmaxLayout for UnitTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        local_pos
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        1u32
    }
}

#[cube]
impl<E: Numeric> FragmentMask for UnitTile<E> {
    type Layout = UnitTileLayout;
}
