//! The launchable arguments an operand rides — [`StridedTileArg`] (strided) and [`TmaTileArg`]
//! (tensor map) — plus their constructors and the [`Storage`]/quantization plan config
//! they carry. `tile()` turns each into an in-kernel [`Tile`](crate::Tile).

use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantParam, QuantScheme, QuantStore};
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
    /// How stages derived from this operand are laid out and cooperatively filled.
    pub stage: StagePlan,
}

impl Storage {
    /// Every axis tiled, no passthrough; `levels` read off the tensor's rank.
    pub fn of(physical_rank: usize, logical_rank: usize) -> Self {
        Storage {
            start_axis: 0,
            levels: physical_rank / logical_rank - 1,
            check_bounds: false,
            stage: StagePlan::default(),
        }
    }

    pub fn passthrough(start_axis: usize, levels: usize) -> Self {
        Storage {
            start_axis,
            levels,
            check_bounds: false,
            stage: StagePlan::default(),
        }
    }

    /// Set whether edge reads/writes must be bounds-checked.
    pub fn checked(mut self, check_bounds: bool) -> Self {
        self.check_bounds = check_bounds;
        self
    }

    /// Set the stage layout the derived stages take.
    pub fn staged(mut self, layout: StageStorage) -> Self {
        self.stage.layout = layout;
        self
    }

    /// Set the launch's cube size (units per cube).
    pub fn units(mut self, units: usize) -> Self {
        self.stage.units = units;
        self
    }
}

/// The strided [`Delivery`]'s argument: a [`VecTensor`] plus its comptime line
/// [`vector_size`](Self::vector_size), [`Space`], [`Storage`] and load knobs;
/// [`tile`](StridedTileArg::tile) turns it into a `Tile` in-kernel. For a quantized operand,
/// `E` is the storage element and [`tile_dequant`](StridedTileArg::tile_dequant) picks the
/// served type.
#[derive(CubeType, CubeLaunch)]
pub struct StridedTileArg<'a, E: Numeric> {
    pub tensor: &'a VecTensor<E>,
    /// Physical vectorization (`Vector<E, vector_size>` line size) of the operand's contiguous
    /// innermost axis; `1` is scalar. Always equals the binding's width
    /// ([`MemData::from_tensor`] asserts it).
    #[cube(comptime)]
    pub vector_size: usize,
    #[cube(comptime)]
    pub space: Space,
    /// The buffer's physical mapping plus the [`StagePlan`] its derived stages take.
    #[cube(comptime)]
    pub storage: Storage,
    /// Quantization side-channel, `None` for a plain operand (every constructor's default;
    /// [`quantized`](StridedTileArgLaunch::quantized) opts in).
    pub quant: ComptimeOption<QuantArg>,
}

#[cube]
impl<'a, E: Numeric> StridedTileArg<'a, E> {
    /// Serve the tensor's own element type. The plain path; a quantized operand goes
    /// through [`tile_dequant`](Self::tile_dequant) to name its served type.
    pub fn tile(&self) -> Tile<E> {
        if comptime!(self.quant.is_some()) {
            panic!(
                "StridedTileArg::tile: a quantized operand is served via StridedTileArg::tile_dequant"
            )
        }
        MemData::from_tensor(
            self.tensor,
            comptime!(self.vector_size),
            comptime!(self.space.clone()),
            comptime!(self.storage),
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
            ComptimeOption::Some(q) => {
                ComptimeOption::new_Some(QuantInfo::native(q, comptime!(self.space.rank())))
            }
            ComptimeOption::None => ComptimeOption::new_None(),
        };
        MemData::<O>::from_tensor_quant::<E>(
            self.tensor,
            comptime!(self.vector_size),
            comptime!(self.space.clone()),
            comptime!(self.storage),
            quant,
        )
    }
}

/// The quantization side-channel of a [`StridedTileArg`]: the scale grid plus the comptime
/// [`QuantScheme`] that says how to fold it back in. Optional on the arg so the *same* kernel runs
/// quantized or not (the tile dequantizes on read).
#[derive(CubeType, CubeLaunch)]
pub struct QuantArg {
    /// One scale per block, on the scheme's own block grid; per-tensor is the single-scale
    /// degenerate case. Read as `f32` straight through, so the scheme's param must say so.
    pub scales: OwnedTensor<f32>,
    #[cube(comptime)]
    pub scheme: QuantScheme,
}

/// The TMA [`Delivery`]'s argument: the tensor-map [`ViewMut`] carrier (the descriptor
/// owns the box, the [`TmaDynLayout`] the coordinate rules) plus the comptime [`Space`].
/// [`StridedTileArg`]'s twin, since a `CubeLaunch` type cannot hold both a `&Tensor` and a
/// tensor map. Built by [`TmaTileArgLaunch::tensor_map`](crate::TmaTileArgLaunch::tensor_map).
#[derive(CubeType, CubeLaunch)]
pub struct TmaTileArg<E: Numeric> {
    pub view: ViewMut<'static, E, CoordsDyn>,
    #[cube(comptime)]
    pub space: Space,
}

#[cube]
impl<E: Numeric> TmaTileArg<E> {
    /// Serve the tensor map as a [`TmaGmem`](crate::TileKind::TmaGmem) tile: not
    /// element-addressable, its only sink is a hardware bulk copy into shared memory.
    pub fn tile(&self) -> Tile<E> {
        TmaData::from_tensor_map(self.view.clone(), comptime!(self.space.clone()))
    }
}

impl<E: Numeric, R: Runtime> StridedTileArgLaunch<'static, E, R> {
    /// Start describing a strided tile kernel argument sourced from `binding`: a
    /// [`StridedTileSource`] builder. Set the required [`space`](StridedTileSource::space) and
    /// [`subspace`](StridedTileSource::subspace) (`build` won't compile until both are set), then
    /// optionally [`batches`](StridedTileSource::batches), [`levels`](StridedTileSource::levels),
    /// [`vectorize`](StridedTileSource::vectorize), or [`checked`](StridedTileSource::checked).
    /// Optional defaults are the safe ones, so a forgotten optional setter degrades
    /// performance, never correctness.
    pub fn source<'a>(binding: TensorBinding<R>) -> StridedTileSource<'a, Unset, Unset, E, R> {
        StridedTileSource::new(binding)
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
        units: usize,
    ) -> Self {
        Self::strided(
            binding.into_tensor_arg(),
            v,
            space.project(&layout.distinct_axes()),
            Storage::from(layout).checked(check).units(units),
        )
    }

    /// Load a strided global tensor as a tile served in `vector_size`-wide lines (the binding is
    /// typed `Vector<E, vector_size>`, see [`VecTensor`](crate::VecTensor)). Its
    /// `[pre…, grid…, tile…]` buffer is tiled in-kernel over `space` (the [`Tile`](crate::Tile) reads
    /// the physical shape/strides off the tensor). The [`Storage`] carries the tiling depth and the
    /// overhang bounds-check.
    pub fn strided(
        tensor: TensorArg<R>,
        vector_size: usize,
        space: Space,
        mut storage: Storage,
    ) -> Self {
        // Default the stage layout from the space; `units` rides in on `storage` (a
        // `Launcher` stamped it), and [`stage`](Self::stage) can still override the layout.
        storage.stage.layout = StageStorage::for_space(&space);
        Self::new(
            VecTensorArg::new(tensor, vector_size),
            vector_size,
            space,
            storage,
            ComptimeOptionArgs::None,
        )
    }

    /// Override the derived stages' [`StageStorage`] layout (default
    /// [`StageStorage::for_space`]).
    pub fn stage(mut self, stage: StageStorage) -> Self {
        self.storage.stage.layout = stage;
        self
    }

    /// Mark the operand as quantized: its tensor holds the storage element, and `scales` +
    /// `scheme` let reads dequantize into the kernel's served type
    /// ([`tile_dequant`](crate::StridedTileArg::tile_dequant)). Panics on a scheme this operand
    /// cannot serve.
    pub fn quantized(mut self, scales: TensorArg<R>, scheme: QuantScheme) -> Self {
        validate_scheme(&self.space, self.vector_size, scheme);
        // `vector_size` names the *served* width throughout; the binding is typed at the
        // *storage* width, so re-bind the tensor as what it physically is — packed storage
        // (a plain binding again for native's factor of 1). This is the only seam that knows
        // the scheme, so the re-binding lives here rather than on every caller.
        self.tensor = VecTensorArg::packed(
            self.tensor.into_tensor(),
            self.vector_size,
            scheme.num_quants(),
        );
        self.quant = ComptimeOptionArgs::Some(QuantArgLaunch::new(scales, scheme));
        self
    }
}

/// Reject a [`QuantScheme`] this operand cannot serve, at launch and on the caller's thread. Every
/// rule here is also an in-kernel assumption, but a kernel-side assert fires on a device thread,
/// where it reads as zeroed output rather than as a rejection — so this is the one gate.
///
/// A tile reads a scale as its window's own start plus the block index *within* the window
/// ([`ScaleLayout`]), which is the true block only if no window straddles a block edge. Every
/// window is a level's cut, and its origin is a multiple of that cut, so per axis each level's
/// edge must tile whole blocks or fit inside one. A line is one read, so it may not straddle
/// either. Per-tensor's block edges are `1` and divide everything.
fn validate_scheme(space: &Space, vector_size: usize, scheme: QuantScheme) {
    // `Native` holds one element per value; `PackedU32` carries `num_quants` of them per `u32`,
    // which the view unpacks on read. A packed store must pack along the innermost (contiguous,
    // vectorized) axis — that is the one whose lanes the view lays down contiguously. Sub-byte
    // native stores aren't wired.
    match scheme.store {
        QuantStore::Native => {}
        QuantStore::PackedU32(dim) => {
            assert!(
                dim == 0,
                "StridedTileArgLaunch::quantized: a packed-u32 operand must pack along the \
                 innermost axis (dim 0), got {dim}"
            );
            assert!(
                vector_size.is_multiple_of(scheme.num_quants()),
                "StridedTileArgLaunch::quantized: the innermost axis is served in \
                 {vector_size}-wide lines, which must be a multiple of the {}-value packing \
                 factor, else a line splits a u32",
                scheme.num_quants()
            );
        }
        other => panic!(
            "StridedTileArgLaunch::quantized: quantization storage {other:?} is not supported \
             (native or packed-u32)"
        ),
    }
    // The scales ride a `f32` buffer ([`QuantArg`]) read straight through, so a narrower param
    // would reinterpret its bytes.
    assert!(
        scheme.param == QuantParam::F32,
        "StridedTileArgLaunch::quantized: scales are read as f32, got {:?}",
        scheme.param
    );

    let rank = space.rank();
    let block = block_edges(scheme, rank);
    let inner = block[rank - 1];
    assert!(
        inner.is_multiple_of(vector_size),
        "StridedTileArgLaunch::quantized: the innermost axis is served in {vector_size}-wide \
         lines, which its {inner}-element scale blocks must be a multiple of, else one line \
         straddles two scales"
    );

    // Every window is some level's cut, so the final space (which carries no cut) has nothing
    // left to check: its extents are the last level's edges.
    let mut level = space.clone();
    while !level.is_final() {
        for (p, axis) in level.axes().enumerate() {
            let (edge, block) = (level.partitioner().edge(axis), block[p]);
            assert!(
                edge.is_multiple_of(block) || block.is_multiple_of(edge),
                "StridedTileArgLaunch::quantized: {axis:?} is cut into {edge}-element tiles, \
                 which straddle its {block}-element scale blocks; a tile must cover whole blocks \
                 or sit inside one"
            );
        }
        level = level.divide();
    }
}

impl<E: Numeric, R: Runtime> TmaTileArgLaunch<E, R> {
    /// Load a TMA tensor-map as a tile argument. `dims` is the operand's logical runtime
    /// `(batch, rows, cols)`; `transposed` flags a col-major operand whose descriptor
    /// swapped its inner pair (the layout swaps coords back). `space` is the operand's
    /// already-projected tile space: rank 3 with a leading batch axis, rank 2 without.
    pub fn tensor_map(
        tensor_map: TensorMapArg<R, Tiled>,
        space: Space,
        dims: (u32, u32, u32),
        transposed: bool,
    ) -> Self {
        let batched = match space.rank() {
            2 => false,
            3 => true,
            r => panic!(
                "TmaTileArg: the descriptor is (batch, row, col); rank {r} space unsupported"
            ),
        };
        let layout = TmaDynLayoutLaunch::new(dims, batched, transposed);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::new(view, space)
    }
}

/// In-kernel tensor-map layout: aligns the operand's logical [`CoordsDyn`] to the
/// descriptor's 3-D `(batch, row, col)`. A batchless (rank-2) operand gets batch `0`, a
/// unit batch broadcasts; a col-major (`transposed`) descriptor swapped its inner pair, so
/// the layout swaps coords back. `shape()` stays logical, so a tile's `bound` aligns with
/// its space whatever the descriptor order. Same rules as cubek-matmul's legacy
/// `SimpleTmaGlobalLayout` (tuple coords); keep the two in step.
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
