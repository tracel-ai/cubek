//! The addressable backing store ([`MemData`], gmem and smem): its layouts, windows, and
//! the memory-side [`Tile`] operations (construction, views, the cooperative copy).

use cubecl::{
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, View, ViewMut,
        layout::{Coords1d, CoordsDyn, Layout, LayoutExpand, tiled_view::TiledLayout},
    },
};

use crate::*;

/// The lifetime-erased buffer plus the physical shape/strides and tiling spec to
/// rebuild its [`GmemLayout`]. Fixed at construction, so a staged smem sub-tile keeps
/// addressing its whole buffer after [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemData<T: Numeric> {
    /// Backing store, scalar-typed by Rust-side erasure only: the real binding/alloc element is
    /// `Vector<T, vector_size>` (see [`VecTensor`](crate::VecTensor)), so re-grouping to lines
    /// at that width is a no-op.
    pub(crate) buffer: Box<[T]>,
    /// Physical line size (`Vector<T, vector_size>`) of the backing store, `1` when
    /// unvectorized; held comptime so `size!` can read it.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    /// Accumulates across [`at`](Tile::at)s.
    origin: CoordsDyn,
    extent: CoordsDyn,
    /// Absolute logical extent per axis (the valid region); `origin + pos` beyond it is
    /// the partial-tile overhang. Preserved across [`at`](Tile::at), unlike `extent`.
    pub(crate) bound: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    /// Tiled axes, each split into `levels + 1` `[grid…, tile…]` parts.
    #[cube(comptime)]
    num_tiled: usize,
    /// `0` = smem / untiled.
    #[cube(comptime)]
    levels: usize,
    /// Whether edge reads/writes must be bounds-checked. Always `false` for smem (it
    /// never overhangs); gmem inherits its operand's launch-time flag.
    #[cube(comptime)]
    check: bool,
    /// Present when the buffer physically holds quantized data (see [`QuantInfo`]): reads through
    /// [`Tile::flat`] dequantize into `T`; every other element view refuses the tile.
    pub(crate) quant: ComptimeOption<QuantInfo>,
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// Wrap a launched [`VecTensor`] into a whole `Gmem` tile. Shape and strides come in
    /// scalar-unit and convert here to *line-unit* (the buffer indexes in lines): the contiguous
    /// innermost axis counts lines, coarser strides divide by `w`; the launcher gates `w > 1`
    /// on divisibility.
    pub fn from_tensor(
        tensor: &VecTensor<T>,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] storage: Storage,
    ) -> Tile<T> {
        MemData::<T>::from_tensor_quant::<T>(
            tensor,
            vector_size,
            space,
            storage,
            ComptimeOption::new_None(),
        )
    }

    /// [`from_tensor`](MemData::from_tensor) from a storage-typed tensor: the buffer physically
    /// holds `I` while the tile serves `T`, dequantizing on read per `quant`. The plain path is
    /// `I == T` with `quant == None`; [`TileArg::tile_dequant`] is the kernel-side constructor.
    pub fn from_tensor_quant<I: Numeric>(
        tensor: &VecTensor<I>,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] storage: Storage,
        quant: ComptimeOption<QuantInfo>,
    ) -> Tile<T> {
        let tensor_vector_size = tensor.vector_size();
        comptime!(assert!(
            tensor_vector_size == vector_size,
            "MemData::from_tensor: comptime vector_size differs from the binding's width"
        ));

        let start_axis = comptime!(storage.start_axis);
        let num_tiled = comptime!(space.rank() - storage.start_axis);
        let levels = comptime!(storage.levels);
        let rank = comptime!(start_axis + (levels + 1) * num_tiled);
        let last = comptime!(rank - 1);
        let w = comptime!(vector_size as u32);
        let mut physical_shape = CoordsDyn::new();
        let mut physical_strides = CoordsDyn::new();
        #[unroll]
        for i in 0..rank {
            let extent = tensor.shape(i) as u32;
            let stride = tensor.stride(i) as u32;
            if comptime!(i == last) {
                // Innermost (contiguous, scalar stride 1): count lines; consecutive lines
                // are one line apart.
                physical_shape.push(extent / w);
                physical_strides.push(stride);
            } else {
                // Coarser axes re-express their scalar strides in lines.
                physical_shape.push(extent);
                physical_strides.push(stride / w);
            }
        }
        // Re-typing the buffer to the served `T` is only a static coercion; a quantized
        // store truly holds `I` and the read view downcasts back (`flat_storage`).
        let buffer = unsafe {
            tensor
                .as_slice()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        };
        // Logical bound folded from the physical shape, so it's correct for tiled
        // operands too (the physical buffer is padded; the logical extent is not).
        let bound = logical_bound(&physical_shape, start_axis, num_tiled, levels);
        // The whole-tile window. A `Dynamic` axis takes its runtime size from `bound`, so the
        // top-level extent never bakes into the kernel; a `Static` axis keeps its comptime size.
        let (origin, extent) = top_window(comptime!(space.clone()), &bound, vector_size);
        Tile::<T> {
            tile_kind: TileKind::new_Gmem(MemData::<T> {
                buffer,
                vector_size: comptime!(vector_size),
                physical_shape,
                physical_strides,
                origin,
                extent,
                bound,
                start_axis,
                num_tiled,
                levels,
                check: comptime!(storage.check_bounds),
                quant,
            }),
            space: comptime!(space),
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of
    /// `operand`, at the same physical width.
    pub fn smem_like(operand: &Tile<T>) -> Tile<T> {
        MemData::smem(comptime!(operand.space.divide()), operand.vector_size())
    }

    /// Allocate a shared-memory tile over `space`, at physical `vector_size` (the slice is
    /// allocated natively wide, then scalar-erased). A stage bound for the cmma instruction is
    /// storage-tiled at the final tile (one contiguous block per fragment) so the cmma
    /// transaction reads it unstrided; anything else is plain row-major.
    pub fn smem(#[comptime] space: Space, #[comptime] vector_size: usize) -> Tile<T> {
        let levels =
            comptime!((!space.is_final() && space.partitioner().leaf().is_cmma()) as usize);
        let size!(W) = vector_size;
        let smem = Shared::<[Vector<T, W>]>::new_slice(comptime!(space.tile_size() / vector_size));
        let buffer = unsafe {
            smem.inner_ref()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        };
        let (physical_shape, physical_strides) =
            storage_layout(comptime!(space.clone()), vector_size, levels);
        let (origin, extent) = full_window(comptime!(space.clone()), vector_size);
        // Smem never overhangs its own buffer, so the bound is the extent and checks are off.
        let bound = extent.clone();
        Tile::<T> {
            tile_kind: TileKind::new_Smem(MemData::<T> {
                buffer,
                vector_size,
                physical_shape,
                physical_strides,
                origin,
                extent,
                bound,
                start_axis: comptime!(0usize),
                num_tiled: comptime!(space.rank()),
                levels,
                check: comptime!(false),
                quant: ComptimeOption::new_None(),
            }),
            space: comptime!(space),
        }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A read [`View`] over `Vector<T, W>` lines: the scalar buffer re-grouped into its physical
    /// width, then re-viewed through the base layout and [`Window`]. `W` is the line width
    /// (`self.vector_size`); pass `Const<1>` when only the (width-invariant) leading shape is needed.
    pub fn view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) => {
                if comptime!(g.quant.is_some()) {
                    panic!(
                        "Tile::view: a quantized tile only serves dequantized reads (Tile::flat)"
                    )
                }
                g.lines::<W>().view(g.base()).view(g.window())
            }
            TileKind::Smem(g) => g.lines::<W>().view(g.base()).view(g.window()),
            TileKind::TmaGmem(_) => panic!("Tile::view: a tma source has no element view"),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::view: a cmma fragment has no memory view")
            }
        }
    }

    pub fn view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match &mut self.tile_kind {
            TileKind::Gmem(g) => {
                if comptime!(g.quant.is_some()) {
                    panic!("Tile::view_mut: writing a quantized tile requires requantization")
                }
                let base = g.base();
                let window = g.window();
                g.lines_mut::<W>().view_mut(base).view_mut(window)
            }
            TileKind::Smem(g) => {
                let base = g.base();
                let window = g.window();
                g.lines_mut::<W>().view_mut(base).view_mut(window)
            }
            TileKind::TmaGmem(_) => panic!("Tile::view_mut: a tma source has no element view"),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::view_mut: a cmma fragment has no memory view")
            }
        }
    }
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// Memory transport leaf: cooperative cyclic copy of `src` into `self`, whole
    /// `Vector<T, W>` lines at `self`'s width, unit `u` moving lines `u`, `u + CUBE_DIM`, ….
    /// The caller owns the rendezvous: a `sync_cube` must separate this fill from its readers.
    pub(crate) fn fill_from(&mut self, src: &MemData<T>) {
        let size!(W) = comptime!(self.vector_size);
        let s = src.flat_transparent::<T, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            // `src` zeroes reads past its logical bound (the partial-tile overhang); the
            // staged buffer is unchecked, so the full padded cell is still written.
            d.write(i, s.read(i));
            i += workers;
        }
    }

    /// This buffer's byte length (its length is in native lines, so widened by the vector
    /// size): the transaction count a TMA fill into it lands.
    pub(crate) fn size_bytes(&self) -> u32 {
        self.buffer.len() as u32 * comptime!(T::type_size() as u32 * self.vector_size as u32)
    }

    /// The base layout: the `[grid…, tile…]` split (gmem, `levels > 0`) or a plain
    /// strided dot (smem, `levels = 0`).
    fn base(&self) -> GmemLayout {
        GmemLayout {
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            start_axis: self.start_axis,
            num_tiled: self.num_tiled,
            levels: self.levels,
        }
    }

    fn window(&self) -> Window {
        Window::new(self.origin.clone(), self.extent.clone(), self.bound.clone())
    }

    /// The window extent, for shape-only readers that must not regroup the buffer.
    pub(crate) fn extent(&self) -> CoordsDyn {
        self.extent.clone()
    }

    /// The buffer re-grouped into `Vector<T, W>` lines, which the line-unit base/window
    /// layouts address. `W` is the width the buffer already has, so the regroup is a
    /// no-op; only the cmma row stride widens back to scalars ([`row_stride`](MemData::row_stride)).
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.buffer.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](MemData::lines).
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.buffer.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// [`lines`](MemData::lines) with the buffer re-typed to the quantized storage
    /// element `I` it truly holds (see [`QuantInfo`]).
    fn lines_storage<I: Numeric, W: Size>(&self) -> &[Vector<I, W>] {
        let storage = unsafe { self.buffer.downcast_unchecked::<I>() };
        storage.as_vectorized().with_vector_size::<W>()
    }

    /// The buffer from this window's origin on: the base a cmma load/store addresses,
    /// rows stepping by the scalar [`row_stride`](MemData::row_stride) (cmma takes a line
    /// slice with a scalar stride). Requires an unmasked store whose window doesn't split
    /// rows across storage tiles.
    pub(crate) fn window_slice(&self) -> &[T] {
        let offset = self.window_offset();
        self.buffer.slice(offset, self.buffer.len())
    }

    /// The mutable twin of [`window_slice`](MemData::window_slice).
    pub(crate) fn window_slice_mut(&mut self) -> &mut [T] {
        let offset = self.window_offset();
        let end = self.buffer.len();
        self.buffer.slice_mut(offset, end)
    }

    /// Line offset of the window origin: the origin through the base layout. On a tiled
    /// store the window must lie within one storage tile.
    fn window_offset(&self) -> usize {
        comptime!(assert!(
            !self.check,
            "MemData::window_offset: cmma cannot mask an overhang"
        ));
        self.base().to_source_pos(self.origin.clone())
    }

    /// Scalar stride between matrix rows: the line-unit physical stride of the leaf
    /// tile's row axis, widened back to scalars.
    pub(crate) fn row_stride(&self) -> u32 {
        let rows = comptime!(self.start_axis + (self.levels + 1) * self.num_tiled - 2);
        self.physical_strides[rows] * comptime!(self.vector_size as u32)
    }

    /// Re-view this buffer through `layout` as a [`MatrixView`], carrying its own `check` flag
    /// so the leaf masks without being asked.
    pub(crate) fn masked<W: Size>(&self, layout: BatchMatrix) -> MatrixView<'_, Vector<T, W>> {
        if comptime!(self.quant.is_some()) {
            panic!("Tile::matrix: a quantized tile only serves dequantized reads (Tile::flat)")
        }
        MaskedView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window())
                .view(layout),
            comptime!(self.check),
        )
    }

    /// The mutable twin of [`masked`](MemData::masked).
    pub(crate) fn masked_mut<W: Size>(
        &mut self,
        layout: BatchMatrix,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        if comptime!(self.quant.is_some()) {
            panic!("Tile::matrix_mut: writing a quantized tile requires requantization")
        }
        let base = self.base();
        let window = self.window();
        let check = comptime!(self.check);
        MaskedViewMut::new(
            self.lines_mut::<W>()
                .view_mut(base)
                .view_mut(window)
                .view_mut(layout),
            check,
        )
    }

    /// Re-view this buffer as a flat 1-D [`FlatView`] over its [`Window`] extent,
    /// carrying the `check` flag so a flat scan masks the overhang without being asked.
    pub(crate) fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        FlatView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window())
                .view(FlatLayout::new(self.extent.clone())),
            comptime!(self.check),
        )
    }

    /// [`flat`](MemData::flat) over the storage element `I` a quantized buffer truly holds; the
    /// [`QuantizedView`](crate::QuantizedView) wraps it to dequantize each read.
    pub(crate) fn flat_storage<I: Numeric, W: Size>(&self) -> FlatView<'_, Vector<I, W>> {
        FlatView::new(
            self.lines_storage::<I, W>()
                .view(self.base())
                .view(self.window())
                .view(FlatLayout::new(self.extent.clone())),
            comptime!(self.check),
        )
    }

    /// Quantization-transparent [`flat`](MemData::flat): a plain store serves the bare
    /// `Direct` read, a quantized one re-types to the storage element `I` and dequantizes
    /// each read into `T`. `#[comptime]`, so the plain path pays nothing.
    pub(crate) fn flat_transparent<I: Numeric, W: Size>(&self) -> TileView<'_, T, I, W, Coords1d> {
        #[comptime]
        match &self.quant {
            ComptimeOption::Some(info) => TileView::new_Quantized(QuantizedView::new(
                self.flat_storage::<I, W>(),
                T::cast_from(info.scale),
                comptime!(info.scheme),
            )),
            ComptimeOption::None => TileView::new_Direct(self.flat::<W>()),
        }
    }

    /// The mutable twin of [`flat`](MemData::flat).
    pub(crate) fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        if comptime!(self.quant.is_some()) {
            panic!("Tile::flat_mut: writing a quantized tile requires requantization")
        }
        let base = self.base();
        let window = self.window();
        let extent = self.extent.clone();
        let check = comptime!(self.check);
        FlatViewMut::new(
            self.lines_mut::<W>()
                .view_mut(base)
                .view_mut(window)
                .view_mut(FlatLayout::new(extent)),
            check,
        )
    }

    /// The `i`-th batch matrix as a 2-D view. Mirrors [`Tile::matrix_mut`] for callers that
    /// hold the tile-kind rather than the whole tile, so the `space` is passed in.
    pub(crate) fn matrix_mut<W: Size>(
        &mut self,
        i: usize,
        #[comptime] space: Space,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        let rank = comptime!(space.rank());
        let rows = comptime!(space.extent_at(rank - 2));
        // The `Space` is scalar; `cols` counts lines, so divide the innermost extent by the width.
        let cols = comptime!(space.extent_at(rank - 1) / self.vector_size);
        // Leading (batch) extents are width-invariant; the window extent is the view's shape.
        let shape = self.extent();
        let mut batches = CoordsDyn::new();
        #[unroll]
        for p in 0..rank - 2 {
            let mut weight = 1;
            #[unroll]
            for q in comptime!(p + 1)..rank - 2 {
                weight *= shape[q];
            }
            batches.push((i as u32 / weight) % shape[p]);
        }
        self.masked_mut::<W>(BatchMatrix::new(batches, rows, cols))
    }

    /// Window down to `region`: shift the origin by the region's tile coordinate times
    /// the sub-tile edge, crop each axis to that edge, re-box the same buffer. `bound`
    /// is carried through unchanged, so the leaf masks correctly at any nesting depth.
    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> MemData<T> {
        let mut origin = CoordsDyn::new();
        let mut extent = CoordsDyn::new();

        let last = comptime!(space.rank() - 1);
        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            // The innermost (vectorized) axis's edge is a line count, so `/ width`.
            let edge = comptime!(if p == last {
                space.partitioner().edge(axis) / self.vector_size
            } else {
                space.partitioner().edge(axis)
            });
            let index = region.coord(axis);

            origin.push(self.origin[p] + (index * edge) as u32);
            extent.push(edge as u32);
        }

        MemData::<T> {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            vector_size: comptime!(self.vector_size),
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            origin,
            extent,
            bound: self.bound.clone(),
            start_axis: comptime!(self.start_axis),
            num_tiled: comptime!(self.num_tiled),
            levels: comptime!(self.levels),
            check: comptime!(self.check),
            quant: self.quant.clone(),
        }
    }
}

/// The operand's logical extent per axis, folded from its physical `[pre…, grid…, tile…]`
/// shape: passthrough axes pass through, each tiled axis multiplies its per-level factors.
/// Reduces to `physical_shape` for an untiled (strided) operand.
#[cube]
fn logical_bound(
    physical_shape: &CoordsDyn,
    #[comptime] start_axis: usize,
    #[comptime] num_tiled: usize,
    #[comptime] levels: usize,
) -> CoordsDyn {
    let mut bound = CoordsDyn::new();
    #[unroll]
    for i in 0..start_axis {
        bound.push(physical_shape[i]);
    }
    #[unroll]
    for a in 0..num_tiled {
        let mut prod = 1u32;
        #[unroll]
        for l in 0..comptime!(levels + 1) {
            prod *= physical_shape[comptime!(start_axis) + l * num_tiled + a];
        }
        bound.push(prod);
    }
    bound
}

/// The whole-tile window: `origin = 0`, `extent =` the space's per-axis extents. `Space` is
/// conceptual; the innermost (vectorized) axis's extent is a line count, `/ vector_size`.
#[cube]
fn full_window(#[comptime] space: Space, #[comptime] vector_size: usize) -> (CoordsDyn, CoordsDyn) {
    let mut origin = CoordsDyn::new();
    let mut extent = CoordsDyn::new();
    let last = comptime!(space.rank() - 1);

    #[unroll]
    for p in 0..space.rank() {
        origin.push(0);
        let e = comptime!(space.extent(space.axis_at(p)));
        extent.push(comptime!(if p == last { e / vector_size } else { e }) as u32);
    }

    (origin, extent)
}

/// [`full_window`] for the top gmem tile, where an axis may be [`Dynamic`](crate::Extent): such
/// an axis reads its runtime size from `bound` (the folded logical extent) instead of a comptime
/// constant, so the problem shape never specializes the kernel.
#[cube]
fn top_window(
    #[comptime] space: Space,
    bound: &CoordsDyn,
    #[comptime] vector_size: usize,
) -> (CoordsDyn, CoordsDyn) {
    let mut origin = CoordsDyn::new();
    let mut extent = CoordsDyn::new();
    let last = comptime!(space.rank() - 1);

    #[unroll]
    for p in 0..space.rank() {
        origin.push(0);
        let axis = comptime!(space.axis_at(p));
        // The innermost (vectorized) axis is a line count, `/ vector_size`. A `Dynamic` axis reads
        // its size from `bound`, already lined from the physical shape.
        let size = match comptime!(space.extent_raw(axis)) {
            Extent::Static(e) => {
                (comptime!(if p == last { e / vector_size } else { e }) as u32).runtime()
            }
            Extent::Dynamic => bound[p],
        };
        extent.push(size);
    }

    (origin, extent)
}

/// The smem physical shape/strides over `space`, line-unit like [`Tile::from_tensor`].
/// `levels == 0` is a plain row-major buffer; `levels == 1` is the `[grid…, tile…]` storage
/// tiling at the space's final tile, each tile a contiguous block.
#[cube]
fn storage_layout(
    #[comptime] space: Space,
    #[comptime] vector_size: usize,
    #[comptime] levels: usize,
) -> (CoordsDyn, CoordsDyn) {
    // The physical line extents, comptime: `[extents…]` flat, or `[grid…, tile…]`.
    let extents = comptime! {
        let rank = space.rank();
        let mut extents = Vec::new();
        match levels {
            0 => {
                for p in 0..rank {
                    extents.push(space.extent_at(p));
                }
            }
            1 => {
                let fin = space.final_space();
                for p in 0..rank {
                    let (e, t) = (space.extent_at(p), fin.extent_at(p));
                    assert!(
                        e.is_multiple_of(t),
                        "MemData::smem: the final tile must divide the staged space"
                    );
                    extents.push(e / t);
                }
                for p in 0..rank {
                    extents.push(fin.extent_at(p));
                }
            }
            _ => panic!("MemData::smem: one storage-tiling level at most"),
        }
        let last = extents.len() - 1;
        extents[last] /= vector_size;
        extents
    };

    let mut shape = CoordsDyn::new();
    let mut strides = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(extents.len()) {
        shape.push(comptime!(extents[p]) as u32);
        let weight = comptime!(extents[p + 1..].iter().product::<usize>());
        strides.push(weight as u32);
    }

    (shape, strides)
}

/// In-kernel twin of cubecl's `TiledViewLayout`, which has no in-kernel
/// constructor. Splits each logical axis into its `[grid…, tile…]` parts
/// ([`TiledLayout`]) then dots the physical strides.
#[derive(CubeType, Clone)]
pub struct GmemLayout {
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    #[cube(comptime)]
    num_tiled: usize,
    #[cube(comptime)]
    levels: usize,
}

#[cube]
impl Layout for GmemLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let split = TiledLayout::new(
            self.physical_shape.clone(),
            self.start_axis,
            self.num_tiled,
            self.levels,
        );

        let physical = split.to_source_pos(pos);

        let mut offset = 0;

        #[unroll]
        for i in 0..self.physical_strides.len() {
            offset += physical[i] * self.physical_strides[i];
        }

        offset as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        let split = TiledLayout::new(
            self.physical_shape.clone(),
            self.start_axis,
            self.num_tiled,
            self.levels,
        );

        split.shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;

        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }

        valid
    }
}

/// The layout [`Tile::at`] applies: shift every axis to `origin` and crop it to
/// `extent`. Same rank as the source; the rank-reducing 2-D slice is
/// [`BatchMatrix`](super::BatchMatrix).
#[derive(CubeType, Clone)]
pub struct Window {
    origin: CoordsDyn,
    extent: CoordsDyn,
    /// Absolute logical extent (the valid region). `shape()` stays `extent` (the tile
    /// cell, so loops cover the whole padded tile), but `is_in_bounds` clips against
    /// `bound` so a checked read/write zeroes / skips the overhang.
    bound: CoordsDyn,
}

#[cube]
impl Window {
    pub fn new(origin: CoordsDyn, extent: CoordsDyn, bound: CoordsDyn) -> Self {
        Window {
            origin,
            extent,
            bound,
        }
    }
}

#[cube]
impl Layout for Window {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for i in 0..self.origin.len() {
            out.push(self.origin[i] + pos[i]);
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.clone()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;

        // The cell can overhang the matrix; a position is valid only if its absolute
        // coordinate (`origin + pos`) is within the logical `bound`.
        #[unroll]
        for i in 0..self.bound.len() {
            valid = valid && self.origin[i] + pos[i] < self.bound[i];
        }

        valid
    }
}
