use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::pipeline::{LocalTile, LocalTileLayout, Mask, MaskExpand, UnitTile, UnitTileLayout};
use crate::tile::{InnerLayout, StridedTile};

/// Layout of an attention-style mask fragment across the units of a plane.
/// Replaces what was previously the `SoftmaxLayout` trait + per-impl config.
/// Purely comptime — all variants carry only comptime data.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum MaskLayout {
    /// Each unit owns a full row-major copy of the tile.
    Unit(UnitTileLayout),
    /// The tile is fragmented across plane units, with the layout described by
    /// [`LocalTileLayout`].
    Local(LocalTileLayout),
}

impl MaskLayout {
    pub const fn unit(num_rows: u32, num_cols: u32) -> MaskLayout {
        MaskLayout::Unit(UnitTileLayout {
            num_rows,
            num_cols,
            transposed_load: false,
        })
    }

    pub const fn local(
        tile_shape: Coords2d,
        plane_dim: u32,
        inner_layout: InnerLayout,
    ) -> MaskLayout {
        let total_elements = tile_shape.0 * tile_shape.1;
        let elements_per_unit = total_elements.div_ceil(plane_dim);
        let (num_rows_per_unit, num_cols_per_unit) = match inner_layout {
            InnerLayout::Contiguous => (1u32, elements_per_unit),
            InnerLayout::SplitRows => (2u32, elements_per_unit / 2u32),
        };
        let unit_size = (num_rows_per_unit, num_cols_per_unit);
        let num_units_per_row = tile_shape.1 / unit_size.1;

        MaskLayout::Local(LocalTileLayout {
            total_size: tile_shape,
            unit_size,
            num_units_per_row,
            plane_dim,
        })
    }
}

#[cube]
/// Returns how many units in a plane participate in the same row.
/// Comptime helper paralleling the previous `SoftmaxLayout::num_units_per_row`.
pub fn mask_layout_num_units_per_row(#[comptime] layout: MaskLayout) -> comptime_type!(u32) {
    match layout {
        MaskLayout::Unit(_) => 1u32,
        MaskLayout::Local(l) => comptime!(l.total_size.1 / l.unit_size.1),
    }
}

#[cube]
/// Maps a per-unit `(row, col)` to its absolute position within the tile.
pub fn mask_layout_absolute_pos(
    #[comptime] layout: MaskLayout,
    local_pos: Coords2d,
) -> Coords2d {
    match layout {
        MaskLayout::Unit(_) => local_pos,
        MaskLayout::Local(l) => {
            let abs_row_index = {
                let row_0 = UNIT_POS_X / l.num_units_per_row;
                let row_jump = comptime!(l.plane_dim / l.num_units_per_row);
                local_pos.0 * row_jump + row_0
            };
            let abs_col_index = l.unit_size.1 * (UNIT_POS_X % l.num_units_per_row) + local_pos.1;
            (abs_row_index, abs_col_index)
        }
    }
}

/// Concrete fragment used to materialize a mask. Mirrors the [`Tile`] shape
/// distinction: `Unit` for register-resident tiles, `Local` for cmma-bounced
/// tiles that go through shared memory.
#[derive(CubeType)]
pub enum MaskFragment<E: Numeric> {
    Unit(UnitTile<E>),
    Local(LocalTile<E>),
}

#[cube]
impl<E: Numeric> MaskFragment<E> {
    pub fn new(#[comptime] layout: MaskLayout) -> MaskFragment<E> {
        match layout {
            MaskLayout::Unit(l) => MaskFragment::new_Unit(UnitTile::<E>::new(comptime!(l))),
            MaskLayout::Local(l) => MaskFragment::new_Local(LocalTile::<E>::new(comptime!(l))),
        }
    }

    pub fn load_from_strided_tile<E2: Numeric, ES: Size>(
        &mut self,
        tile: &StridedTile<E2, ES>,
    ) {
        match self {
            MaskFragment::Unit(t) => t.load_from_strided_tile::<E2, ES>(tile),
            MaskFragment::Local(t) => t.load_from_strided_tile::<E2, ES>(tile),
        }
    }
}

#[cube]
impl<E: Numeric> Mask for MaskFragment<E> {
    fn should_mask(&self, local_pos: Coords2d) -> bool {
        match self {
            MaskFragment::Unit(t) => t.should_mask(local_pos),
            MaskFragment::Local(t) => t.should_mask(local_pos),
        }
    }
}
