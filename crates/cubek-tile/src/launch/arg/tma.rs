//! A TMA tensor map as an operand's values ([`TileArgLaunch::tensor_map`]) and its in-kernel
//! layouts.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    layout::{CoordsDyn, Layout, LayoutExpand},
    view::launch::ViewArg,
};

use crate::*;

/// The box a TMA descriptor moves: logical `(rows, cols)`, optional batch, and whether the
/// descriptor is transposed (column-major).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmaBox {
    pub rows: u32,
    pub cols: u32,
    pub batch: Option<u32>,
    pub transposed: bool,
}

/// What the TMA family launches from: the tensor map, the axes it spans, and its box.
pub struct TmaOperand {
    pub map: TensorMapArg<Tiled>,
    pub axes: Vec<Axis>,
    pub shape: TmaBox,
}

impl TmaOperand {
    /// The tensor map as an operand's values, read at `E`, under no scales.
    pub fn arg<E: Numeric>(self) -> TileArgLaunch<'static, E> {
        TileArgLaunch::tensor_map(self.map, &self.axes, self.shape)
    }
}

impl<E: Numeric> TileArgLaunch<'static, E> {
    /// A TMA tensor map as a tile argument over `axes`: two, or three with `shape.batch` set.
    pub fn tensor_map(tensor_map: TensorMapArg<Tiled>, axes: &[Axis], shape: TmaBox) -> Self {
        let layout = Self::dyn_layout(axes, shape, 1);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::of_view(view, TileSpec::direct(axes))
    }

    /// [`tensor_map`](Self::tensor_map) over a packed operand: the descriptor moves the words its
    /// values are stored in, `field.per_word()` values a cell along its innermost dimension, which
    /// the layout divides the values' coordinate along it by. `shape` counts values.
    pub fn tensor_map_packed(
        tensor_map: TensorMapArg<Tiled>,
        axes: &[Axis],
        shape: TmaBox,
        field: Field,
    ) -> Self {
        let layout = Self::dyn_layout(axes, shape, field.per_word() as u32);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::of_view(view, TileSpec::direct(axes).packed(field))
    }

    /// The layout of a descriptor over `axes` moving cells of `per_cell` values along its
    /// innermost dimension.
    fn dyn_layout(axes: &[Axis], shape: TmaBox, per_cell: u32) -> TmaDynLayoutLaunch {
        let batched = match (axes.len(), shape.batch) {
            (2, None) => false,
            (3, Some(_)) => true,
            (r, batch) => panic!(
                "TileArgLaunch::tensor_map: the descriptor is (batch, row, col); {r} axes with \
                 batch {batch:?} is not a matrix nor a batched one"
            ),
        };
        let dims = (shape.batch.unwrap_or(1), shape.rows, shape.cols);
        TmaDynLayoutLaunch::new(dims, batched, shape.transposed, per_cell)
    }

    /// A storage-tiled operand's tensor map over two `axes`, landing as `tiles.landing` says.
    pub fn tensor_map_stored(
        tensor_map: TensorMapArg<Tiled>,
        axes: &[Axis],
        tiles: TmaTiles,
    ) -> Self {
        Self::of_view(
            Self::stored_view(tensor_map, axes, tiles),
            TileSpec::direct(axes),
        )
    }

    /// [`tensor_map_stored`](Self::tensor_map_stored) over a packed operand: its words hold
    /// `field`, `field.per_word()` values a word along `cols`. `tiles` counts values.
    pub fn tensor_map_stored_packed(
        tensor_map: TensorMapArg<Tiled>,
        axes: &[Axis],
        tiles: TmaTiles,
        field: Field,
    ) -> Self {
        let view = Self::stored_view(tensor_map, axes, tiles);
        Self::of_view(view, TileSpec::direct(axes).packed(field))
    }

    fn stored_view(
        tensor_map: TensorMapArg<Tiled>,
        axes: &[Axis],
        tiles: TmaTiles,
    ) -> ViewArg<CoordsDyn> {
        assert_eq!(
            axes.len(),
            2,
            "TileArgLaunch::tensor_map_stored: a storage-tiled descriptor is a matrix; batch it by \
             launching per batch"
        );
        let layout = TmaStoredLayoutLaunch::new(tiles.dims, tiles.tile, tiles.landing);
        ViewArg::new_tensor_map_tiled::<TmaStoredLayout>(tensor_map, layout)
    }
}

/// Where a box of a storage-tiled operand lands its values: the order the descriptor states the
/// operand in (`cubek::std::launch::tma::tma_operand_tiled`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmaLanding {
    /// One storage tile a box, landing as the contiguous run it is stored as: the descriptor is
    /// the operand as stored, `[.., R/tr, C/tc, tr, tc]`.
    Tiles,
    /// Whole tiles both ways, landing the box's rows one after another, each a run of its tiles'
    /// rows, as a row-major operand's box would land them: the descriptor is the operand as
    /// `[R/tr, tr, C/tc, tc]`, the tiles' rows nested inside the tiles' columns.
    Rows,
}

/// What a storage-tiled descriptor moves: the operand's logical `(rows, cols)`, its storage tile
/// `(rows, cols)`, and where a box lands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmaTiles {
    pub dims: (u32, u32),
    pub tile: (u32, u32),
    pub landing: TmaLanding,
}

/// In-kernel tensor-map layout splitting a logical `(row, col)` into storage-tile coordinates.
#[derive(CubeType, CubeLaunch, Clone)]
pub(crate) struct TmaStoredLayout {
    /// Logical `(rows, cols)` of the operand.
    dims: (u32, u32),
    /// The storage tile `(rows, cols)`.
    tile: (u32, u32),
    /// The order the descriptor states the tiles in.
    #[cube(comptime)]
    landing: TmaLanding,
}

#[cube]
impl Layout for TmaStoredLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (tile_rows, tile_cols) = self.tile;
        let mut src = CoordsDyn::new();
        // A box origin is tile-aligned, so the in-tile offset is zero.
        match comptime!(self.landing) {
            TmaLanding::Tiles => {
                src.push(pos[0] / tile_rows);
                src.push(pos[1] / tile_cols);
                src.push(0u32);
                src.push(0u32);
            }
            TmaLanding::Rows => {
                src.push(pos[0] / tile_rows);
                src.push(0u32);
                src.push(pos[1] / tile_cols);
                src.push(0u32);
            }
        }
        src
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        // TMA loads are clamped by the descriptor; no in-kernel bounds check.
        (self.to_source_pos(pos), true)
    }

    fn shape(&self) -> Self::Coordinates {
        let (rows, cols) = self.dims;
        let mut s = CoordsDyn::new();
        s.push(rows);
        s.push(cols);
        s
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true
    }
}

/// In-kernel tensor-map layout aligning logical [`CoordsDyn`] to the descriptor's
/// `(batch, row, col)`.
#[derive(CubeType, CubeLaunch, Clone)]
pub(crate) struct TmaDynLayout {
    /// Logical `(batch, rows, cols)` of the operand.
    dims: (u32, u32, u32),
    #[cube(comptime)]
    batched: bool,
    #[cube(comptime)]
    transposed: bool,
    /// Values one cell of the descriptor's innermost dimension holds: one, or a packed field's
    /// values to a word.
    #[cube(comptime)]
    per_cell: u32,
}

#[cube]
impl Layout for TmaDynLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (batch, _rows, _cols) = self.dims;
        let mut src = CoordsDyn::new();
        if comptime!(self.batched) {
            // A unit-batch descriptor is a broadcast: always read batch 0.
            src.push(select(batch == 1, 0u32, pos[0]));
        } else {
            src.push(0u32);
        }
        let (r, c) = comptime!(if self.batched { (1, 2) } else { (0, 1) });
        // TMA discards the last stride, so a col-major descriptor is transposed; swap back. The
        // innermost dimension counts cells, each `per_cell` values.
        let per_cell = comptime!(self.per_cell);
        if comptime!(self.transposed) {
            src.push(pos[c]);
            src.push(pos[r] / per_cell);
        } else {
            src.push(pos[r]);
            src.push(pos[c] / per_cell);
        }
        src
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        // TMA loads are clamped by the descriptor; no in-kernel bounds check.
        (self.to_source_pos(pos), true)
    }

    fn shape(&self) -> Self::Coordinates {
        let (batch, rows, cols) = self.dims;
        let mut s = CoordsDyn::new();
        if comptime!(self.batched) {
            s.push(batch);
        }
        s.push(rows);
        s.push(cols);
        s
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true
    }
}
