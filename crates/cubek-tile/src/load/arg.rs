//! The launchable arguments an operand rides — [`TileArg`] (strided) and [`TmaArg`]
//! (tensor map) — plus their constructors and the [`Storage`]/quantization plan config
//! they carry. `tile()` turns each into an in-kernel [`Tile`](crate::Tile).

use cubecl::prelude::*;
use cubecl::quant::scheme::QuantScheme;
use cubecl::std::tensor::{
    ViewMut,
    layout::{CoordsDyn, Layout, LayoutExpand},
    view::launch::ViewArg,
};

use crate::*;

/// How a launched tensor's `[pre…, grid…, tile…]` buffer maps to the logical
/// [`Space`]. A property of the tensor, distinct from the space's partitioner.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Storage {
    pub start_axis: usize,
    pub levels: usize,
    /// Whether this operand's logical extent can overhang its tile grid, so edge
    /// reads/writes must be bounds-checked. Set from divisibility at launch; `false`
    /// keeps the unchecked (divisible) fast path.
    pub check_bounds: bool,
}

impl Storage {
    /// Every axis tiled, no passthrough; `levels` read off the tensor's rank.
    pub fn of(physical_rank: usize, logical_rank: usize) -> Self {
        Storage {
            start_axis: 0,
            levels: physical_rank / logical_rank - 1,
            check_bounds: false,
        }
    }

    pub fn passthrough(start_axis: usize, levels: usize) -> Self {
        Storage {
            start_axis,
            levels,
            check_bounds: false,
        }
    }

    /// Set whether edge reads/writes must be bounds-checked.
    pub fn checked(mut self, check_bounds: bool) -> Self {
        self.check_bounds = check_bounds;
        self
    }
}

/// The strided [`Delivery`]'s argument: a scalar `&Tensor` plus its comptime line
/// [`vector_size`](Self::vector_size), [`Space`], [`Storage`] and load knobs;
/// [`tile`](TileArg::tile) turns it into a `Tile` in-kernel. For a quantized operand,
/// `E` is the storage element and [`tile_dequant`](TileArg::tile_dequant) picks the
/// served type.
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<E>,
    /// Physical vectorization (`Vector<E, vector_size>` line size) of the operand's contiguous
    /// innermost axis; `1` is scalar.
    #[cube(comptime)]
    pub vector_size: usize,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub storage: Storage,
    /// Storage layout of the smem stages derived from this operand.
    #[cube(comptime)]
    pub stage: StageStorage,
    /// Quantization side-channel, `None` for a plain operand (every constructor's default;
    /// [`quantized`](TileArgLaunch::quantized) opts in).
    pub quant: ComptimeOption<QuantArg>,
}

#[cube]
impl<'a, E: Numeric> TileArg<'a, E> {
    /// Serve the tensor's own element type. The plain path; a quantized operand goes
    /// through [`tile_dequant`](Self::tile_dequant) to name its served type.
    pub fn tile(&self) -> Tile<E> {
        if comptime!(self.quant.is_some()) {
            panic!("TileArg::tile: a quantized operand is served via TileArg::tile_dequant")
        }
        MemData::from_tensor(
            self.tensor,
            comptime!(self.vector_size),
            comptime!(self.space.clone()),
            comptime!(self.storage),
            comptime!(self.stage),
        )
    }

    /// Serve `O` from a storage-typed operand: `quant = Some` attaches the scale + scheme so reads
    /// dequantize `E → O` transparently; `quant = None` is the plain path (the launch binds
    /// `E == O`). For kernels that thread both types via `#[define]` and run quantized or not.
    pub fn tile_dequant<O: Numeric>(&self) -> Tile<O> {
        // `#[comptime]`: whether the operand is quantized is a trace-time fact, so the match
        // resolves at expand and the plain path pays nothing.
        let quant = #[comptime]
        match &self.quant {
            // Per-tensor native: a single scale at flat index 0.
            ComptimeOption::Some(q) => ComptimeOption::new_Some(QuantInfo {
                scale: q.scales[0],
                scheme: comptime!(q.scheme),
            }),
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        MemData::<O>::from_tensor_quant::<E>(
            self.tensor,
            comptime!(self.vector_size),
            comptime!(self.space.clone()),
            comptime!(self.storage),
            comptime!(self.stage),
            quant,
        )
    }
}

/// The quantization side-channel of a [`TileArg`]: the scale grid plus the comptime
/// [`QuantScheme`] that says how to fold it back in. Optional on the arg so the *same* kernel runs
/// quantized or not (the tile dequantizes on read).
#[derive(CubeType, CubeLaunch)]
pub struct QuantArg {
    /// Per-tensor scales (currently a single value at flat index 0).
    pub scales: OwnedTensor<f32>,
    #[cube(comptime)]
    pub scheme: QuantScheme,
}

/// The TMA [`Delivery`]'s argument: the tensor-map [`ViewMut`] carrier plus the comptime
/// [`Space`] and box geometry. [`TileArg`]'s twin — a `CubeLaunch` type cannot hold both
/// a `&Tensor` and a tensor map, so each family has its own argument and
/// [`DeliveryFamily`] lets one kernel body take either. Built by
/// [`TmaArgLaunch::tensor_map`](crate::TmaArgLaunch::tensor_map).
#[derive(CubeType, CubeLaunch)]
pub struct TmaArg<E: Numeric> {
    pub view: ViewMut<'static, E, CoordsDyn>,
    #[cube(comptime)]
    pub space: Space,
    /// The descriptor's box (what one `tensor_map_load` copies), in descriptor order.
    #[cube(comptime)]
    pub box_rows: u32,
    #[cube(comptime)]
    pub box_cols: u32,
    /// Whether the descriptor swapped its inner pair (a col-major operand).
    #[cube(comptime)]
    pub transposed: bool,
}

#[cube]
impl<E: Numeric> TmaArg<E> {
    /// Serve the tensor map as a [`TmaGmem`](crate::TileKind::TmaGmem) tile: not
    /// element-addressable, its only sink is a hardware bulk copy into shared memory.
    pub fn tile(&self) -> Tile<E> {
        TmaData::from_tensor_map(
            self.view.clone(),
            comptime!(self.space.clone()),
            comptime!(self.box_rows),
            comptime!(self.box_cols),
            comptime!(self.transposed),
        )
    }
}

impl<E: Numeric, R: Runtime> TileArgLaunch<'static, E, R> {
    /// Start describing a strided tile kernel argument sourced from `binding`: a
    /// [`TileSource`] builder. Set the required [`space`](TileSource::space) and
    /// [`subspace`](TileSource::subspace) (`build` won't compile until both are set), then
    /// optionally [`batches`](TileSource::batches), [`levels`](TileSource::levels),
    /// [`vectorize`](TileSource::vectorize), or [`checked`](TileSource::checked).
    /// Optional defaults are the safe ones, so a forgotten optional setter degrades
    /// performance, never correctness.
    pub fn source<'a>(binding: TensorBinding<R>) -> TileSource<'a, Unset, Unset, E, R> {
        TileSource::new(binding)
    }

    /// Load a strided operand from its realized [`ConcreteLayout`]: derive the spanned
    /// axes and the tiling [`Storage`] from the layout, and project `space` onto those
    /// axes. The innermost axis is served as `Vector<E, v>` lines, re-lined in-kernel so
    /// the scalar buffer's shape/strides pass through untouched.
    pub fn from_concrete(
        binding: TensorBinding<R>,
        layout: &ConcreteLayout,
        space: &Space,
        v: usize,
        check: bool,
    ) -> Self {
        Self::strided(
            binding.into_tensor_arg(),
            v,
            space.project(&layout.distinct_axes()),
            Storage::from(layout).checked(check),
        )
    }

    /// Load a strided global tensor as a tile served in `vector_size`-wide lines. Its
    /// `[pre…, grid…, tile…]` buffer is tiled in-kernel over `space` (the [`Tile`](crate::Tile) reads
    /// the physical shape/strides off the tensor). The [`Storage`] carries the tiling depth and the
    /// overhang bounds-check.
    pub fn strided(
        tensor: TensorArg<R>,
        vector_size: usize,
        space: Space,
        storage: Storage,
    ) -> Self {
        let stage = StageStorage::for_space(&space);
        Self::new(
            tensor,
            vector_size,
            space,
            storage,
            stage,
            ComptimeOptionArgs::None,
        )
    }

    /// Override the derived stages' [`StageStorage`] layout (default
    /// [`StageStorage::for_space`]).
    pub fn stage(mut self, stage: StageStorage) -> Self {
        self.stage = stage;
        self
    }

    /// Mark the operand as quantized: its tensor holds the storage element, and `scales` +
    /// `scheme` let reads dequantize into the kernel's served type
    /// ([`tile_dequant`](crate::TileArg::tile_dequant)).
    pub fn quantized(mut self, scales: TensorArg<R>, scheme: QuantScheme) -> Self {
        self.quant = ComptimeOptionArgs::Some(QuantArgLaunch::new(scales, scheme));
        self
    }
}

impl<E: Numeric, R: Runtime> TmaArgLaunch<E, R> {
    /// Load a TMA tensor-map as a tile argument, [`TileArg`](crate::TileArg)'s delivery twin.
    /// The hardware bulk-copies the descriptor's box per `tensor_map_load`; `box_shape` is
    /// that box in *logical* `(rows, cols)`. `dims` is the operand's logical
    /// `(batch, rows, cols)` (runtime, so shapes never specialize the kernel) and
    /// `transposed` flags a col-major operand whose descriptor swapped its inner pair (the
    /// layout swaps coords back). `space` is the operand's already-projected tile space:
    /// rank 3 with a leading batch axis, rank 2 without.
    pub fn tensor_map(
        tensor_map: TensorMapArg<R, Tiled>,
        space: Space,
        dims: (u32, u32, u32),
        box_shape: (u32, u32),
        transposed: bool,
    ) -> Self {
        let batched = match space.rank() {
            2 => false,
            3 => true,
            r => panic!("TmaArg: the descriptor is (batch, row, col); rank {r} space unsupported"),
        };
        let (box_rows, box_cols) = box_shape;
        let layout = TmaDynLayoutLaunch::new(dims, batched, transposed);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::new(view, space, box_rows, box_cols, transposed)
    }
}

/// In-kernel tensor-map layout: aligns the operand's logical [`CoordsDyn`] to the
/// descriptor's 3-D `(batch, row, col)`. A batchless (rank-2) operand gets batch `0`, a
/// unit batch broadcasts; a col-major (`transposed`) descriptor swapped its inner pair, so
/// the layout swaps coords back. `shape()` stays logical, so a tile's `bound` aligns with
/// its space whatever the descriptor order.
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
