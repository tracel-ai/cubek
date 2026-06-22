//! The tile-loading API: the one place a launched tensor or TMA tensor-map becomes a
//! [`TileArgLaunch`]. Every client (matmul, dequantize, …) loads tiles through these two
//! constructors — strided and TMA — so the carrier and delivery wiring lives here, not at each
//! call site.

use cubecl::prelude::*;
use cubecl::std::tensor::{
    launch::ViewArg,
    layout::{
        CoordsDyn, Layout, LayoutExpand,
        tiled_view::{TileSpec, TiledViewLayout},
    },
};

use crate::{Axis, ConcreteLayout, Delivery, Space, Storage, TileArgLaunch};

/// A realized physical layout maps straight to a tile [`Storage`]: its passthrough (batch) prefix
/// is `start_axis`, its storage-tiling depth is `levels`.
impl From<&ConcreteLayout> for Storage {
    fn from(layout: &ConcreteLayout) -> Self {
        Storage::passthrough(layout.passthrough(), layout.levels())
    }
}

impl<E: Numeric, V: Size, R: Runtime> TileArgLaunch<E, V, R> {
    /// Load a strided operand from its realized [`ConcreteLayout`]: the tiling [`Storage`] comes from
    /// the layout, the innermost (`cols`) axis is lined by `v`, and `space` is projected onto the
    /// operand's `axes`. `axes` is the binding's *logical* dim order (`[batches…, row, col]`) — the
    /// `ConcreteLayout` is physical (major-to-minor) and a col-major operand's physical order differs,
    /// so it can't supply the projection order, only the (order-free) tiling depth.
    pub fn from_concrete(
        mut binding: TensorBinding<R>,
        layout: &ConcreteLayout,
        axes: &[Axis],
        space: &Space,
        v: usize,
        check: bool,
    ) -> Self {
        // Re-line the buffer as `Vector<E, v>`: the contiguous innermost stride stays 1, every
        // coarser stride and the `cols` extent shrink by `v` (a no-op at `v == 1`, e.g. tiled).
        let n = binding.strides.len();
        let mut shape = binding.shape.to_vec();
        let mut strides = binding.strides.to_vec();
        shape[n - 1] /= v;
        for s in &mut strides[..n - 1] {
            *s /= v;
        }
        binding.shape = shape[..].into();
        binding.strides = strides[..].into();

        Self::strided(
            binding.into_tensor_arg(),
            space.project(axes),
            Storage::from(layout).checked(check),
        )
    }

    /// Load a strided global tensor as a tile. Its `[pre…, grid…, tile…]` buffer is addressed by a
    /// `TiledViewLayout` over `space` (the layout reads the physical shape/strides off the tensor),
    /// retiring any in-kernel stride math. The [`Storage`] carries the tiling depth and the
    /// overhang bounds-check.
    pub fn strided(tensor: TensorArg<R>, space: Space, storage: Storage) -> Self {
        let spec = TileSpec {
            start_axis: storage.start_axis as u8,
            num_tiled: space.rank() - storage.start_axis,
            levels: storage.levels,
        };
        let view = ViewArg::new_tensor::<TiledViewLayout>(tensor, spec);
        Self::new(view, space, Delivery::Strided(storage))
    }

    /// Load a TMA tensor-map as a tile. The hardware bulk-copies a `rows × cols` box per
    /// `tensor_map_load`; `shape` is the descriptor's `(batch, rows, cols)` and `transposed` flags
    /// a col-major operand (whose descriptor swapped its inner pair — the view swaps coords back).
    pub fn tma(
        tensor_map: TensorMapArg<R, Tiled>,
        space: Space,
        shape: (u32, u32, u32),
        transposed: bool,
    ) -> Self {
        let (_batch, rows, cols) = shape;
        let layout = TmaDynLayoutLaunch::new(shape, transposed);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::new(
            view,
            space,
            Delivery::Tma {
                rows,
                cols,
                transposed,
            },
        )
    }
}

/// In-kernel tensor-map layout over [`CoordsDyn`] `(batch, row, col)`: broadcasts a unit batch and
/// swaps `(row, col)` for a col-major (`transposed`) descriptor. The dynamic-coordinate counterpart
/// of matmul's `SimpleTmaGlobalLayout`, so a tensor-map view shares the strided path's `CoordsDyn`.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct TmaDynLayout {
    /// `(batch, rows, cols)` of the descriptor (already box-swapped for `transposed`).
    shape: (u32, u32, u32),
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl Layout for TmaDynLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, coords: CoordsDyn) -> CoordsDyn {
        let (batch, _rows, _cols) = self.shape;
        // A unit-batch descriptor is a broadcast: always read batch 0.
        let b = select(batch == 1, 0u32, coords[0]);
        let mut out = CoordsDyn::new();
        out.push(b);
        // TMA discards the last stride, so a col-major descriptor is transposed; swap back.
        if comptime!(self.transposed) {
            out.push(coords[2]);
            out.push(coords[1]);
        } else {
            out.push(coords[1]);
            out.push(coords[2]);
        }
        out
    }

    fn to_source_pos_checked(&self, coords: CoordsDyn) -> (CoordsDyn, bool) {
        (self.to_source_pos(coords), true.runtime())
    }

    fn shape(&self) -> CoordsDyn {
        let (batch, rows, cols) = self.shape;
        let mut s = CoordsDyn::new();
        s.push(batch);
        s.push(rows);
        s.push(cols);
        s
    }

    fn is_in_bounds(&self, _pos: CoordsDyn) -> bool {
        // TMA loads are clamped by the descriptor; no in-kernel bounds check.
        true.runtime()
    }
}
