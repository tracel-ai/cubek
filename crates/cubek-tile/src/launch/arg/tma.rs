//! The TMA delivery's argument: a tensor map with its comptime [`TileSpec`], and the in-kernel
//! layouts that align its coordinates with the descriptor.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    ViewMut,
    layout::{CoordsDyn, Layout, LayoutExpand},
    view::launch::ViewArg,
};

use crate::*;

/// The TMA [`Delivery`]'s argument: the tensor-map [`ViewMut`] carrier (the descriptor owns the
/// box, the [`TmaDynLayout`] the coordinate rules) with its comptime [`TileSpec`]; [`TileArg`]'s
/// twin. Built by [`TmaTileArgLaunch::tensor_map`](crate::TmaTileArgLaunch::tensor_map).
#[derive(CubeType, CubeLaunch)]
pub struct TmaTileArg<E: Numeric> {
    pub view: ViewMut<'static, E, CoordsDyn>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<E: Numeric> TmaTileArg<E> {
    /// Serve the tensor map as a [`TmaGmem`](crate::TileKind::TmaGmem) tile over the
    /// kernel's one `space`; the spec's width and storage don't apply to a tensor map.
    pub fn tile(&self, #[comptime] space: Partitioning) -> Tile<E> {
        let own = comptime!(space.space().subspace(self.spec.axes()));
        let data = TmaData::from_tensor_map(
            self.view.clone(),
            comptime!(own.rank()),
            comptime!(self.spec.units),
        );
        Tile::new(
            TileKind::new_TmaGmem(data),
            comptime!(Placement::root(own.clone(), space.levels().to_vec())),
        )
    }
}

/// The box a TMA descriptor moves and the runtime extents it moves within: the operand's logical
/// `(rows, cols)`, its batch when it has one, and whether the descriptor is column-major (TMA
/// discards the last stride, so a col-major descriptor is transposed and the layout swaps back).
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
    /// A TMA tensor map as a tile argument over `axes`: two for a matrix, three with the batch
    /// leading, which `shape.batch` then states. Width and storage don't apply to a tensor map,
    /// so the spec is built.
    pub fn tensor_map(tensor_map: TensorMapArg<Tiled>, axes: &[Axis], shape: TmaBox) -> Self {
        let batched = match (axes.len(), shape.batch) {
            (2, None) => false,
            (3, Some(_)) => true,
            (r, batch) => panic!(
                "TmaTileArg: the descriptor is (batch, row, col); {r} axes with batch {batch:?} \
                 is not a matrix nor a batched one"
            ),
        };
        let dims = (shape.batch.unwrap_or(1), shape.rows, shape.cols);
        let layout = TmaDynLayoutLaunch::new(dims, batched, shape.transposed);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::new(view, TileSpec::direct(axes))
    }

    /// Load a storage-tiled operand's tensor-map over `axes`; `dims` is its logical `(rows, cols)`.
    /// The descriptor is the stored `[.., R/tr, C/tc, tr, tc]` and its box one storage tile, so
    /// unlike [`tensor_map`](Self::tensor_map) it keeps the stored rank and splits the coordinate.
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

/// In-kernel tensor-map layout for a storage-tiled operand: splits the logical `(row, col)` into
/// the descriptor's `(row / tr, col / tc, row % tr, col % tc)`, as [`Storage::Tiled`] does on the
/// cooperative path. `shape()` stays logical, so a tile's `bound` aligns with its space.
///
/// A load's box origin is tile-aligned (the storage tile is the stage, which the routine
/// enforces), so the inner pair is always `0`.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct TmaStoredLayout {
    /// Logical `(rows, cols)` of the operand.
    dims: (u32, u32),
    /// The storage tile `(rows, cols)`, which is the descriptor's box.
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
        // A box origin is tile-aligned, so the offset inside the tile is structurally zero.
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

/// In-kernel tensor-map layout: aligns the operand's logical [`CoordsDyn`] to the descriptor's 3-D
/// `(batch, row, col)`: a rank-2 operand gets batch `0`, a unit batch broadcasts, a `transposed`
/// descriptor's inner pair swaps back. `shape()` stays logical, so a tile's `bound` fits its space.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct TmaDynLayout {
    /// Logical `(batch, rows, cols)` of the operand.
    dims: (u32, u32, u32),
    #[cube(comptime)]
    batched: bool,
    #[cube(comptime)]
    transposed: bool,
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
        // TMA discards the last stride, so a col-major descriptor is transposed; swap back.
        if comptime!(self.transposed) {
            src.push(pos[c]);
            src.push(pos[r]);
        } else {
            src.push(pos[r]);
            src.push(pos[c]);
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
