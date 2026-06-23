//! The tile-loading API: the one place a launched tensor or TMA tensor-map becomes a
//! [`TileArgLaunch`]. Every client (matmul, dequantize, …) loads tiles through these two
//! constructors — strided and TMA — so the carrier and delivery wiring lives here, not at each
//! call site.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    launch::ViewArg,
    layout::{
        CoordsDyn, Layout, LayoutExpand,
        tiled_view::{TileSpec, TiledViewLayout},
    },
};

use crate::{Axis, ConcreteLayout, Delivery, PhysicalAxis, Space, Storage, TileArgLaunch};

/// A realized physical layout maps straight to a tile [`Storage`]: its passthrough (batch) prefix
/// is `start_axis`, its storage-tiling depth is `levels`.
impl From<&ConcreteLayout> for Storage {
    fn from(layout: &ConcreteLayout) -> Self {
        Storage::passthrough(layout.passthrough(), layout.levels())
    }
}

impl<E: Numeric, V: Size, R: Runtime> TileArgLaunch<E, V, R> {
    /// Start describing a strided tile kernel argument sourced from `binding` — a [`TileSource`]
    /// builder. Set the two required parts — the [`space`](TileSource::space) it projects from and the
    /// [`subspace`](TileSource::subspace) block it iterates (`build` won't compile until both are set) —
    /// then optionally complementary outer [`batches`](TileSource::batches), a
    /// [`vectorize`](TileSource::vectorize) line size, or opt out of the bounds-check
    /// ([`checked`](TileSource::checked)). Optional defaults are the safe ones — scalar, batchless,
    /// checked — so a forgotten *optional* setter degrades performance, never correctness.
    pub fn source<'a>(binding: TensorBinding<R>) -> TileSource<'a, Unset, Unset, E, V, R> {
        TileSource {
            data: TileSourceData {
                binding,
                space: None,
                subspace: &[],
                batch_axes: &[],
                v: 1,
                check: true,
                _ty: PhantomData,
            },
            _state: PhantomData,
        }
    }

    /// Load a strided operand from its realized [`ConcreteLayout`]: derive the spanned axes
    /// ([`distinct_axes`](ConcreteLayout::distinct_axes)) and the tiling [`Storage`] from the layout,
    /// line the innermost (`cols`) axis by `v`, and project `space` onto those axes. The
    /// matmul-agnostic loader — the `layout`'s axes are in the binding's dim order (its `Coordinates`
    /// match the buffer dim-for-dim), so a client just builds the operand's layout and hands it here.
    pub fn from_concrete(
        mut binding: TensorBinding<R>,
        layout: &ConcreteLayout,
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
            space.project(&layout.distinct_axes()),
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

/// Typestate marker: a required [`TileSource`] field has been set.
pub struct Set;
/// Typestate marker: a required [`TileSource`] field is still missing.
pub struct Unset;

/// The fields an [`TileSource`] accumulates; the typestate lives in the wrapper, not here.
struct TileSourceData<'a, E, V, R: Runtime> {
    binding: TensorBinding<R>,
    space: Option<&'a Space>,
    subspace: &'a [Axis],
    batch_axes: &'a [Axis],
    v: usize,
    check: bool,
    _ty: PhantomData<(E, V)>,
}

/// Typestate builder for a strided tile kernel argument, started with [`TileArgLaunch::source`]. The
/// argument occupies a subspace of the global space, named by two complementary axis groups: the
/// inner [`subspace`](Self::subspace) block (the tile it iterates — its trailing buffer dims,
/// storage-tiled so labels repeat level-major: dim `i` is `subspace[i % subspace.len()]`) and the
/// outer [`batches`](Self::batches) (its leading dims, one axis each, dropped when size 1 — numpy
/// broadcast omission). The binding is set at construction; the `Sp`/`Sub` markers track the two
/// remaining required setters, so [`build`](Self::build) exists only once both [`space`](Self::space)
/// and [`subspace`](Self::subspace) are [`Set`]. Borrows the axis slices + `space` for the chain.
pub struct TileSource<'a, Sp, Sub, E, V, R: Runtime> {
    data: TileSourceData<'a, E, V, R>,
    _state: PhantomData<(Sp, Sub)>,
}

impl<'a, Sp, Sub, E, V, R: Runtime> TileSource<'a, Sp, Sub, E, V, R> {
    /// The global iteration space this argument projects from (required).
    pub fn space(mut self, space: &'a Space) -> TileSource<'a, Set, Sub, E, V, R> {
        self.data.space = Some(space);
        TileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The inner block of axes the operand iterates — its `[row, col]` for a matmul (required,
    /// non-empty). Complementary to [`batches`](Self::batches), the outer dims.
    pub fn subspace(mut self, axes: &'a [Axis]) -> TileSource<'a, Sp, Set, E, V, R> {
        self.data.subspace = axes;
        TileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The outer (batch) axes, complementary to the [`subspace`](Self::subspace) block: one per
    /// leading buffer dim, dropped when size 1 (numpy broadcast). Default none (unbatched).
    pub fn batches(mut self, axes: &'a [Axis]) -> Self {
        self.data.batch_axes = axes;
        self
    }

    /// Line the innermost axis as `Vector<E, v>` (default `1`, i.e. scalar). Only valid when that
    /// axis is contiguous.
    pub fn vectorize(mut self, v: usize) -> Self {
        self.data.v = v;
        self
    }

    /// Bounds-check the operand's overhang against `space` (default `true`); pass `false` to skip the
    /// check when the tiling is known to divide evenly.
    pub fn checked(mut self, check: bool) -> Self {
        self.data.check = check;
        self
    }
}

impl<'a, E: Numeric, V: Size, R: Runtime> TileSource<'a, Set, Set, E, V, R> {
    /// Build the operand's [`ConcreteLayout`] from its labeled dims and load it via
    /// [`from_concrete`](TileArgLaunch::from_concrete). Available only once space and subspace are
    /// both set, so the `unwrap` below cannot fire.
    pub fn build(self) -> TileArgLaunch<E, V, R> {
        let TileSourceData {
            mut binding,
            space,
            batch_axes,
            subspace,
            v,
            check,
            ..
        } = self.data;
        let space = space.unwrap();
        let mut phys = Vec::new();
        let mut shape = Vec::new();
        let mut strides = Vec::new();

        for (i, &axis) in batch_axes.iter().enumerate() {
            let extent = binding.shape[i];
            if extent == 1 {
                continue; // broadcast omission: the dim and its axis both drop out
            }
            phys.push(PhysicalAxis::new(axis, extent));
            shape.push(extent);
            strides.push(binding.strides[i]);
        }

        let n = subspace.len();
        let block = binding.shape[batch_axes.len()..]
            .iter()
            .zip(&binding.strides[batch_axes.len()..])
            .enumerate();
        for (i, (&extent, &stride)) in block {
            phys.push(PhysicalAxis::new(subspace[i % n], extent));
            shape.push(extent);
            strides.push(stride);
        }

        binding.shape = shape[..].into();
        binding.strides = strides[..].into();
        TileArgLaunch::from_concrete(binding, &ConcreteLayout::new(&phys), space, v, check)
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
