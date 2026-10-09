//! The TMA delivery's argument and its in-kernel layouts.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    ViewMut,
    layout::{CoordsDyn, Layout, LayoutExpand},
    view::launch::ViewArg,
};

use crate::*;

/// The TMA [`Delivery`]'s argument: a tensor-map [`ViewMut`] with its comptime [`TileSpec`].
#[derive(CubeType, CubeLaunch)]
pub struct TmaTileArg<E: Numeric> {
    pub view: ViewMut<'static, E, CoordsDyn>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<E: Numeric> TmaTileArg<E> {
    /// Serve the tensor map as a TMA-filled tile over `space`.
    pub fn tile(&self, #[comptime] space: Partitioning) -> Tile<E> {
        let own = comptime!(space.space().subspace(self.spec.axes()));
        let data = TmaData::from_tensor_map(
            self.view.clone(),
            comptime!(own.rank()),
            comptime!(self.spec.units),
            comptime!(self.spec.packing),
        );
        Tile::new(
            TileKind::new_TmaGmem(data),
            comptime!(Placement::root(own.clone(), space.levels().to_vec())),
        )
    }
}

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

impl<E: Numeric> TmaTileArgLaunch<E> {
    /// A TMA tensor map as a tile argument over `axes`: two, or three with `shape.batch` set.
    pub fn tensor_map(tensor_map: TensorMapArg<Tiled>, axes: &[Axis], shape: TmaBox) -> Self {
        let layout = Self::dyn_layout(axes, shape, 1);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::new(view, TileSpec::direct(axes))
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
        Self::new(view, TileSpec::direct(axes).packed(field))
    }

    /// The layout of a descriptor over `axes` moving cells of `per_cell` values along its
    /// innermost dimension.
    fn dyn_layout(axes: &[Axis], shape: TmaBox, per_cell: u32) -> TmaDynLayoutLaunch {
        let batched = match (axes.len(), shape.batch) {
            (2, None) => false,
            (3, Some(_)) => true,
            (r, batch) => panic!(
                "TmaTileArg: the descriptor is (batch, row, col); {r} axes with batch {batch:?} \
                 is not a matrix nor a batched one"
            ),
        };
        let dims = (shape.batch.unwrap_or(1), shape.rows, shape.cols);
        TmaDynLayoutLaunch::new(dims, batched, shape.transposed, per_cell)
    }

    /// A storage-tiled operand's tensor map over two `axes`; `dims` is its logical `(rows, cols)`.
    pub fn tensor_map_stored(
        tensor_map: TensorMapArg<Tiled>,
        axes: &[Axis],
        dims: (u32, u32),
        tile: (u32, u32),
    ) -> Self {
        assert_eq!(
            axes.len(),
            2,
            "TmaTileArg: a storage-tiled descriptor is a matrix; batch it by launching per batch"
        );
        let layout = TmaStoredLayoutLaunch::new(dims, tile);
        let view = ViewArg::new_tensor_map_tiled::<TmaStoredLayout>(tensor_map, layout);
        Self::new(view, TileSpec::direct(axes))
    }
}

/// In-kernel tensor-map layout splitting a logical `(row, col)` into storage-tile coordinates.
#[derive(CubeType, CubeLaunch, Clone)]
pub(crate) struct TmaStoredLayout {
    /// Logical `(rows, cols)` of the operand.
    dims: (u32, u32),
    /// The storage tile `(rows, cols)`, the descriptor's box.
    tile: (u32, u32),
}

#[cube]
impl Layout for TmaStoredLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (tile_rows, tile_cols) = self.tile;
        let mut src = CoordsDyn::new();
        src.push(pos[0] / tile_rows);
        src.push(pos[1] / tile_cols);
        // A box origin is tile-aligned, so the in-tile offset is zero.
        src.push(0u32);
        src.push(0u32);
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
