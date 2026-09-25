//! Touching a [`Memory`]: filling one from another (the cooperative copy), the views a leaf
//! reads and writes it through, and [`at`](Memory::at), which windows it down to a region.
//!
//! Every cooperative fill here distributes its elements out over `CUBE_DIM` workers indexed by
//! `UNIT_POS`, assuming every unit of the cube runs it. A walk that sets planes aside to fill its
//! stages ([`Level::filled_by`](crate::Level::filled_by)) breaks that: a wrong answer, not a hang.
//!
//! Such a walk is refused where the two meet, and only a bulk copy may be filled by planes of
//! their own. Closing the gap is a worker offset and count on these loops, taken from the pipeline.

use cubecl::{
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, ErasedTensor, View, ViewMut, WriteOnly,
        layout::{Coordinates, Coords1d, Coords2d, CoordsDyn},
    },
};

use crate::*;
use cubecl::unexpanded;

#[cube]
impl<T: Numeric> Tile<T> {
    /// A read [`View`] over `Vector<T, W>` lines: the scalar buffer re-grouped into its physical
    /// width, then re-viewed through the base layout and [`Window`]. `W` is the line width
    /// (`self.store.vector_size`); `Const<1>` when only the width-invariant leading shape matters.
    pub fn view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        let g = self.mem("view");
        if comptime!(g.store.packing != Packing::Plain) {
            panic!(
                "Tile::view: a packed tile only serves values its read unpacks (Tile::copy_from, \
                 Tile::matrix_packed)"
            )
        }
        g.window_view::<W>(comptime!(Guard::Checked))
    }

    /// Scalars of this stage one unit holds in registers across a contraction
    /// ([`MemData::fetched_scalars`]).
    #[allow(dead_code)] // Reached through its expand, from `pipelined_through_registers`.
    pub(crate) fn fetched_scalars(&self) -> comptime_type!(usize) {
        self.mem("fetched_scalars").fetched_scalars()
    }

    /// This unit's share of filling this stage from `src`, read into `fetched` and not yet
    /// written ([`MemData::fetch_straight`]).
    #[allow(dead_code)] // Reached through its expand, from `pipelined_through_registers`.
    pub(crate) fn fetch_from<W: Size>(&self, src: &Tile<T>, fetched: &mut Array<Vector<T, W>>) {
        let space = comptime!(self.place.space.clone());
        self.mem("fetch_from")
            .fetch_straight(src.mem("fetch_from"), space, fetched);
    }

    /// Write what [`fetch_from`](Tile::fetch_from) read into this stage
    /// ([`MemData::store_fetched`]).
    #[allow(dead_code)] // Reached through its expand, from `pipelined_through_registers`.
    pub(crate) fn store_fetched<W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        self.mem_mut("store_fetched").store_fetched(fetched);
    }

    pub fn view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        let g = self.mem_mut("view_mut");
        if comptime!(g.store.quant.is_some()) {
            panic!("Tile::view_mut: writing a quantized tile requires requantization")
        }
        g.window_view_mut::<W>(comptime!(Guard::Checked))
    }
}

impl<T: Numeric> Memory<T> {
    /// This store landing on its way to a fragment ([`Tile::with_landing`]).
    pub(crate) fn with_landing(self) -> Memory<T> {
        unexpanded!()
    }
}

impl<T: Numeric> MemoryExpand<T> {
    pub(crate) fn __expand_with_landing_method(mut self, _scope: &Scope) -> Self {
        self.lands = true;
        self
    }
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// State what the accumulation being lowered starts from ([`init`](Memory::init)).
    pub(crate) fn set_init_from(&mut self, #[comptime] init_from: InitFrom) {
        comptime!({
            self.init_from = init_from;
        });
    }

    /// Zero this window: whole lines at the store's width; a checked window skips
    /// cells past the logical bound.
    pub(crate) fn zero(&mut self) {
        self.init(T::from_int(0));
    }

    /// Initialize this window with `val`: whole lines at the store's width; a checked window
    /// skips cells past the logical bound.
    pub(crate) fn init(&mut self, val: T) {
        let size!(W) = comptime!(self.store.vector_size);
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        for i in 0..total {
            d.write(i, Vector::<T, W>::cast_from(val));
        }
    }

    /// How far this store's stored form travels ([`DequantAt`]). A plain store answers
    /// [`DequantAt::Load`], served and stored being the same element; a [`packed`](Packing::Packed)
    /// store with no scheme answers [`DequantAt::Read`], since a stage copies its words verbatim.
    // The `let`-then-return is load-bearing, see [`quant_pack`](Memory::quant_pack).
    #[allow(clippy::let_and_return)]
    pub(crate) fn dequant_at(&self) -> comptime_type!(DequantAt) {
        let dequant_at = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => comptime!(info.dequant_at),
            ComptimeOption::None => comptime!(match self.store.packing {
                Packing::Plain => DequantAt::Load,
                _ => DequantAt::Read,
            }),
        };
        dequant_at
    }

    /// How this store's values sit in memory, as stated at construction.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(self.store.packing)
    }

    /// This buffer's byte length, widened by the physical width: the transaction count a TMA fill
    /// into it lands. A packed or quantized buffer widens by the *storage* element and physical
    /// width instead, same line count.
    pub(crate) fn size_bytes(&self) -> u32 {
        let lines = self.store.buffer().len() as u32;
        let wp = comptime!(self.store.packing.physical(self.store.vector_size) as u32);
        match comptime!(self.store.packing) {
            Packing::Plain => lines * T::size().comptime() as u32 * wp,
            Packing::Native => lines * i8::size().comptime() as u32 * wp,
            Packing::Packed { field: _ } => lines * u32::size().comptime() as u32 * wp,
        }
    }

    /// The base layout: the `[grid…, tile…]` split (`levels > 0`) or a plain
    /// strided dot (`levels = 0`).
    pub(crate) fn base(&self) -> BufferLayout {
        self.layout.clone()
    }

    pub(crate) fn window(&self) -> Window {
        self.window.clone()
    }

    /// The window extent, for shape-only readers that must not regroup the buffer.
    pub(crate) fn extent(&self) -> Coords<u32> {
        self.window.extent.clone()
    }

    /// The buffer re-grouped into `Vector<T, W>` lines, which the line-unit base/window layouts
    /// address. `W` is the width the buffer already has, so the regroup is a no-op.
    ///
    /// Buffers only, and only where a *slice* is wanted: every layout-addressed read goes through
    /// [`read_view`](Memory::read_view), which an erased source serves and this cannot.
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.store.buffer().as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](Memory::lines). Buffers only: an erased destination has no
    /// address and so no lines to hand out. [`write_view`](Memory::write_view) is the write path
    /// both backings share.
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.store
            .buffer_mut()
            .as_vectorized_mut()
            .with_vector_size_mut::<W>()
    }

    /// The backing as a [`ViewMut`] addressed by `layout`: the write path, and the only one a
    /// [`WriteCall`](Backing::WriteCall) serves. The layout is the same for every backing; only the
    /// end of the address differs, a store or a call, so every mutable view above composes on it.
    fn write_view<W: Size>(
        &mut self,
        layout: BufferLayout,
    ) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match &mut self.store.backing {
            Backing::Buffer(buffer) => buffer
                .as_vectorized_mut()
                .with_vector_size_mut::<W>()
                .view_mut(layout),
            Backing::WriteCall(destination) => {
                ViewMut::new::<&mut ErasedTensor<T, WriteOnly>, Coords1d>(destination, layout)
            }
            Backing::ReadCall(_) => panic!(
                "Memory::write_view: this tile's backing is read through a call, which is \
                 read-only"
            ),
        }
    }

    /// The backing as a [`View`] addressed by `layout`: the read path, and the only one a
    /// [`ReadCall`](Backing::ReadCall) serves; the mirror of [`write_view`](Memory::write_view).
    ///
    /// The slice-shaped half (dense runs, quantized re-typing, tma maps) is deliberately left out:
    /// none is a view over `Coords1d`, so each keeps saying so through [`Store::buffer`].
    fn read_view<W: Size>(&self, layout: BufferLayout) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.store.backing {
            Backing::Buffer(buffer) => buffer.as_vectorized().with_vector_size::<W>().view(layout),
            Backing::ReadCall(producer) => {
                View::new::<&ErasedTensor<T, ReadOnly>, Coords1d>(producer, layout)
            }
            Backing::WriteCall(_) => panic!(
                "Memory::read_view: this tile's backing is written through a call, and is never \
                 read"
            ),
        }
    }

    /// The window as a view over its own coordinates (`pos` relative to the origin), served at
    /// `T` grouped `W` wide: inside one storage tile ([`Held`](Storage::Contiguous)) the run from
    /// the origin under the storage tile's own strides, otherwise the layout walk under `guard`.
    fn window_view<W: Size>(&self, #[comptime] guard: Guard) -> View<'_, Vector<T, W>, CoordsDyn> {
        match comptime!(self.access.storage) {
            Storage::Contiguous => {
                let start = self.window_start.cast::<usize>();
                let all = self.lines::<W>();
                all.slice(start, all.len()).view(self.contiguous_layout())
            }
            Storage::Strided | Storage::Tiled(_) => self
                .read_view::<W>(self.base())
                .view(self.window().with_guard(guard)),
        }
    }

    /// [`window_view`](Memory::window_view) at the storage element `I` the buffer truly holds,
    /// grouped `WP` wide ([`lines_storage`](Memory::lines_storage)). Buffers only.
    pub(crate) fn window_view_storage<I: Numeric, WP: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> View<'_, Vector<I, WP>, CoordsDyn> {
        match comptime!(self.access.storage) {
            Storage::Contiguous => {
                let start = self.window_start.cast::<usize>();
                let all = self.lines_storage::<I, WP>();
                all.slice(start, all.len()).view(self.contiguous_layout())
            }
            Storage::Strided | Storage::Tiled(_) => self
                .lines_storage::<I, WP>()
                .view(self.base())
                .view(self.window().with_guard(guard)),
        }
    }

    /// The mutable twin of [`window_view`](Memory::window_view). A window inside a storage tile
    /// is always a buffer (it came off a binding's tiling), so its run is sliced where an erased
    /// destination could not be.
    fn window_view_mut<W: Size>(
        &mut self,
        #[comptime] guard: Guard,
    ) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match comptime!(self.access.storage) {
            Storage::Contiguous => {
                let start = self.window_start.cast::<usize>();
                let layout = self.contiguous_layout();
                let all = self.lines_mut::<W>();
                let len = all.len();
                all.slice_mut(start, len).view_mut(layout)
            }
            Storage::Strided | Storage::Tiled(_) => {
                let base = self.base();
                let window = self.window().with_guard(guard);
                self.write_view::<W>(base).view_mut(window)
            }
        }
    }

    /// The layout of a window inside one storage tile, relative to its origin: its own extent,
    /// each coordinate addressed by the stride of its innermost fragment, no digit to split. Sits
    /// over the run from [`window_offset`](Memory::window_offset) on, like a fragment load.
    fn contiguous_layout(&self) -> BufferLayout {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "Memory: a window inside one storage tile reads unmasked; a storage-tiled tensor is \
             padded to whole storage tiles"
        ));
        comptime!(self.layout.rows.assert_in_order(
            "Memory::contiguous_layout",
            "a window inside one storage tile is addressed as one run"
        ));
        let positional = comptime!(self.layout.projection.clone());
        let rank = comptime!(positional.coordinate_rank());
        let mut strides = Coords::<u32>::new();
        #[unroll]
        for c in 0..rank {
            let axis = comptime!(positional.logical_axes()[c]);
            let inner = comptime!(*positional.carriers(axis).last().unwrap());
            strides.push(self.layout.physical_strides.at(inner));
        }
        BufferLayout {
            physical_shape: self.window.extent.clone(),
            physical_strides: strides,
            projection: comptime!(Projection::direct(positional.logical_axes())),
            rows: RowPlacement::InOrder,
        }
    }

    /// [`lines`](Memory::lines) with the buffer re-typed to the quantized storage
    /// element `I` it truly holds (see [`QuantInfo`]).
    pub(crate) fn lines_storage<I: Numeric, W: Size>(&self) -> &[Vector<I, W>] {
        let storage = unsafe { self.store.buffer().downcast_unchecked::<I>() };
        storage.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines_storage`](Memory::lines_storage): where a quant stage's
    /// [`fill_straight`](Memory::fill_straight) writes the packed storage words. `I == T` on a
    /// plain copy, a same-type reinterpret.
    pub(crate) fn lines_storage_mut<I: Numeric, W: Size>(&mut self) -> &mut [Vector<I, W>] {
        let storage = unsafe { self.store.buffer_mut().downcast_mut_unchecked::<I>() };
        storage.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// The window as one dense run of lines: index `i` addresses line `origin + i`, one add and no
    /// layout walk. Legal only on a physically contiguous, row-major window (untiled, unmasked,
    /// unquantized); the comptime-checkable parts assert, contiguity is the caller's guarantee.
    pub(crate) fn dense_lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.assert_dense();
        let all = self.lines::<W>();
        let start = self.window_start.cast::<usize>();
        all.slice(start, all.len())
    }

    /// The mutable twin of [`dense_lines`](Memory::dense_lines).
    pub(crate) fn dense_lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.assert_dense();
        let start = self.window_start.cast::<usize>();
        let all = self.lines_mut::<W>();
        let end = all.len();
        all.slice_mut(start, end)
    }

    /// Refuse a window that is not one dense run of lines: the comptime half of
    /// [`dense_lines`](Memory::dense_lines)'s contract.
    fn assert_dense(&self) {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "Memory::dense_lines: a dense window cannot mask an overhang"
        ));
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "Memory::dense_lines: a storage-tiled window is not dense"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "Memory::dense_lines: a gathered window is not dense (sibling windows overlap)"
        ));
        comptime!(assert!(
            self.store.packing == Packing::Plain,
            "Memory::dense_lines: a packed store is served through its packed views"
        ));
    }

    /// The buffer from this window's origin on: the base a cmma load/store addresses, rows
    /// stepping by the scalar [`row_stride`](Memory::row_stride). Requires an unmasked store
    /// whose window does not split rows across storage tiles.
    pub(crate) fn window_slice(&self) -> &[T] {
        let offset = self.window_offset();
        self.store.buffer().slice(offset, self.store.buffer().len())
    }

    /// The mutable twin of [`window_slice`](Memory::window_slice).
    pub(crate) fn window_slice_mut(&mut self) -> &mut [T] {
        let offset = self.window_offset();
        let end = self.store.buffer().len();
        self.store.buffer_mut().slice_mut(offset, end)
    }

    /// Whether this store was opened with a landing ([`Tile::with_landing`]).
    pub(crate) fn has_landing(&self) -> comptime_type!(bool) {
        comptime!(self.lands)
    }

    /// Line offset of the window origin: the accumulated `window_start`. Addresses the window as
    /// one contiguous region, so on a tiled store it must lie inside one storage tile, which is
    /// what [`Storage`] says of it.
    fn window_offset(&self) -> usize {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "Memory::window_offset: cmma cannot mask an overhang"
        ));
        // Reading a window above its storage tile from a base and a row stride would walk straight
        // through a storage tile boundary and return another tile's cells, silently. The layout
        // walk addresses them correctly; a fragment load cannot, and says so.
        match comptime!(self.access.storage) {
            Storage::Strided => {}
            Storage::Contiguous => {}
            Storage::Tiled(Some(level)) => panic!(
                "Memory::window_offset: this window sits above its storage tile (the tile of \
                 level {level}), spanning several, so it is not one contiguous region; descend \
                 through that level first, or read the operand through its layout"
            ),
            Storage::Tiled(None) => panic!(
                "Memory::window_offset: this operand's storage tile is the tile of no level of \
                 the kernel's nest, so no window is known to lie inside one; read it through its \
                 layout, or stage it"
            ),
        }
        // A raw window addresses a row as a base and a stride, which a swizzled row is not.
        comptime!(
            self.layout
                .rows
                .assert_rows_are_runs("Memory::window_slice")
        );
        // A raw window serves the buffer at the element it was erased to, so a quantized store
        // would hand its stored bytes over as served values. Every other door refuses the same way.
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "Memory::window_slice: a packed store has no raw element window; a fragment \
                 load reads it through Tile::matrix_transparent"
            )
        }
        self.window_start.cast::<usize>()
    }

    /// Scalar stride between matrix rows: the line-unit physical stride of the leaf
    /// tile's row axis, widened back to scalars; a constant on a static store.
    pub(crate) fn row_stride(&self) -> u32 {
        let rank = comptime!(self.layout.projection.physical_rank());
        self.row_stride_at(comptime!(rank - 2))
    }

    /// [`row_stride`](Memory::row_stride) with the row axis stated: the logical position a matrix
    /// reader takes as its rows ([`MatrixAxes::edges`]), the physical one on a direct, untiled
    /// store. Any other keeps its own row: its tile's if storage-tiled, else the dim above a fold.
    pub(crate) fn row_stride_at(&self, #[comptime] row: usize) -> u32 {
        let rank = comptime!(self.layout.projection.physical_rank());
        let row = comptime!(
            if self.projection.is_direct() && !self.layout.projection.is_tiled() {
                row
            } else {
                rank - 2
            }
        );
        self.layout
            .physical_strides
            .at(row)
            .times(comptime!(self.store.vector_size as u32).runtime())
    }

    /// Re-view this buffer through `layout` as a [`Masked`], carrying its own `check` flag
    /// so the leaf masks without being asked. `layout` is a [`TileMatrix`] for the 2-D matmul
    /// leaves and an [`ProjectionInKernel`] for a gathered N-D read.
    pub(crate) fn masked<W: Size, C: Coordinates, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "Tile::matrix: a packed tile only serves values its read unpacks \
                 (Tile::matrix_transparent)"
            )
        }
        Masked::new(
            self.window_view::<W>(guard).view(layout),
            comptime!(guard.checks() && self.access.overhang.masks()),
        )
    }

    /// The mask flag a *write* view is built with: [`Overhang::masks`], plus the one policy a
    /// write cannot honour. [`Boundary::Clamp`] folds an out-of-range coordinate onto the edge
    /// cell, so several logical cells would write the same physical one. Refused, not raced.
    fn write_check(&self) -> comptime_type!(bool) {
        // Whole-operand on purpose, unlike the per-axis mask below it: one clamped axis is enough
        // to fold two distinct cells onto one, so there is no such thing as a partly writable
        // clamped operand.
        comptime!(assert!(
            !self.window.boundaries.contains(&Some(Boundary::Clamp))
                || !self.access.overhang.masks(),
            "Memory: a Boundary::Clamp operand is read-only, a clamped write aliases the edge cell"
        ));
        comptime!(self.access.overhang.masks())
    }

    /// The mutable twin of [`masked`](Memory::masked).
    pub(crate) fn masked_mut<W: Size, C: Coordinates, L: TileLayout<C>>(
        &mut self,
        layout: L,
    ) -> MaskedMut<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("Tile::matrix_mut: writing a packed tile requires repacking")
        }
        let check = self.write_check();
        MaskedMut::new(
            self.window_view_mut::<W>(comptime!(Guard::Checked))
                .view_mut(layout),
            check,
        )
    }

    /// Re-view this buffer as a flat 1-D [`FlatView`] over its [`Window`] extent,
    /// carrying the `check` flag so a flat scan masks the overhang without being asked.
    pub(crate) fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        FlatView::new(
            self.window_view::<W>(comptime!(Guard::Checked))
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// Quantization-transparent [`flat`](Memory::flat): a plain store is read as it stands, a
    /// quantized one re-types to the storage element `I` and pairs it with the scales over the same
    /// window, dequantizing each read into `T`. `#[comptime]`, so the plain path pays nothing.
    pub(crate) fn flat_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
    ) -> FlatView<'_, Vector<T, W>> {
        #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => {
                // The storage view groups at the *physical* width: a packed buffer holds
                // `W / num_quants` elements per served line.
                let values = self
                    .window_view_storage::<I, WP>(comptime!(Guard::Checked))
                    .view(FlatLayout::new(self.window.extent.clone()));
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(self.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(FlatLayout::new(self.window.extent.clone()));
                let dequant = info.dequant_view::<I, WP, T, W, Coords1d>(values, scales);
                FlatView::new(dequant.view(), comptime!(self.access.overhang.masks()))
            }
            // The flat scan reads its own window whole, so it keeps the store's own mask.
            ComptimeOption::None => self.unscaled::<WP, W, Coords1d, FlatLayout>(
                FlatLayout::new(self.window.extent.clone()),
                comptime!(Guard::Checked),
            ),
        }
    }

    /// Quantization-transparent [`masked`](Memory::masked): the windowed twin of
    /// [`flat_transparent`](Memory::flat_transparent). A quantized store re-types to `I` and pairs
    /// it with the scales over the same `layout`, so a leaf reads it straight from gmem.
    pub(crate) fn transparent<
        I: Numeric,
        WP: Size,
        W: Size,
        C: Coordinates + 'static,
        L: TileLayout<C>,
    >(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        #[comptime]
        match &self.store.quant {
            // A quantized view *is* a view: cubecl's decodes on read and answers as `Vector<T, W>`,
            // so both arms hand back the same masked view and no caller learns the difference.
            ComptimeOption::Some(info) => {
                // The storage view groups at the *physical* width: a packed buffer holds
                // `W / num_quants` elements per served line.
                let values = self
                    .window_view_storage::<I, WP>(guard)
                    .view(layout.clone());
                // The scales over this same window: `ScaleLayout` resolves a window coordinate
                // to its block's scale, addressed by the same `layout` as the values, so both
                // answer the same coordinate.
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(self.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(layout);
                let dequant = info.dequant_view::<I, WP, T, W, C>(values, scales);
                Masked::new(
                    dequant.view(),
                    comptime!(guard.checks() && self.access.overhang.masks()),
                )
            }
            ComptimeOption::None => self.unscaled::<WP, W, C, L>(layout, guard),
        }
    }

    /// The scale-free half of [`transparent`](Memory::transparent): a plain store read as it
    /// stands, a packed one unpacked at the read ([`PackedView`]) off its field alone, needing no
    /// scheme or grid. `WP` is the physical line the buffer holds, `W` the served one.
    fn unscaled<WP: Size, W: Size, C: Coordinates + 'static, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        let packing = self.packing();
        match comptime!(packing) {
            Packing::Plain => self.masked::<W, C, L>(layout, guard),
            Packing::Native => panic!(
                "Memory::transparent: a native store with nothing to fold in serves its own \
                 element; bind it as that element and let the contraction cast it"
            ),
            Packing::Packed { field } => {
                let words = self
                    .window_view_storage::<u32, WP>(comptime!(Guard::Checked))
                    .view(layout);
                let values = PackedView::<WP, T, W, C>::new(words, comptime!(field));
                Masked::new(
                    values.view(),
                    comptime!(guard.checks() && self.access.overhang.masks()),
                )
            }
        }
    }

    /// [`transparent`](Memory::transparent) with the storage element resolved from this store's
    /// own [`Packing`]: the one place a packing becomes a storage element, so no reader
    /// re-derives the `i8`/`u32` choice from a bare factor.
    pub(crate) fn packed<W: Size, C: Coordinates + 'static, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        let packing = self.packing();
        let physical = comptime!(packing.physical(self.store.vector_size));
        // The `size!` binding sits in the arm that reads it: hoisted above the match, the width
        // it registers is not the one the arm's call sees, and a packed read silently lands on
        // the wrong field.
        match comptime!(packing) {
            Packing::Plain => {
                let size!(WP) = physical;
                self.transparent::<T, WP, W, C, L>(layout, guard)
            }
            Packing::Native => {
                let size!(WP) = physical;
                self.transparent::<i8, WP, W, C, L>(layout, guard)
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = physical;
                self.transparent::<u32, WP, W, C, L>(layout, guard)
            }
        }
    }

    /// The words a packed store holds, as they lie, over the tile's whole logical box: what a
    /// unit loads its line from, decoded later at the read.
    pub(crate) fn nd_words<WP: Size>(
        &self,
        layout: ProjectionInKernel,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<u32, WP>, CoordsDyn> {
        comptime!(assert!(
            matches!(self.store.packing, Packing::Packed { .. }),
            "Memory::nd_words: only a packed store holds words"
        ));
        let words = self.window_view_storage::<u32, WP>(guard).view(layout);
        Masked::new(
            words,
            comptime!(guard.checks() && self.access.overhang.masks()),
        )
    }

    /// The identity step over this window's physical box: the layout a caller that folds the map
    /// itself reads through ([`nd_split`](Tile::nd_split)). Only the map is dropped: the [`Window`]
    /// owning the boundary sits below either way, keeping the box's bound no caller can fold away.
    ///
    /// The logical box test goes with the map, though, so a view over this masks against the
    /// physical box alone: a position folded from an out-of-range logical coordinate is no longer
    /// caught and reads whatever the window says lives at it. The caller owes in-range coordinates.
    pub(crate) fn physical_box(&self) -> CompactionStep {
        let rank = comptime!(self.projection.physical_rank());
        CompactionStep::new(self.window.extent.clone(), comptime!(vec![1; rank]))
    }

    /// The `i`-th batch matrix of this window, read over the axes `axes` names, through the
    /// operand's own mapping: what every 2-D reader of the tile sees.
    pub(crate) fn batch_matrix(
        &self,
        #[comptime] space: Space,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> ProjectedMatrix {
        // Leading (batch) extents are width-invariant; the window extent is the view's shape.
        let bound = self.extent();
        projected_batch_matrix(
            &bound,
            space,
            comptime!(self.projection.clone()),
            self.map.clone(),
            comptime!(self.store.vector_size),
            axes,
            i,
        )
    }

    /// This window's whole logical box as the `rows x cols` matrix an mma fragment reads,
    /// through the operand's own mapping.
    pub(crate) fn whole_matrix(
        &self,
        #[comptime] space: Space,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> ProjectedMatrix {
        projected_whole_matrix(
            space,
            comptime!(self.projection.clone()),
            self.map.clone(),
            comptime!(self.store.vector_size),
            rows,
            cols,
        )
    }

    /// The operand's [`Projection`] applied to this window's logical box: the N-D read surface,
    /// one coordinate per axis of `space`.
    pub(crate) fn axis_projection(&self, #[comptime] space: Space) -> ProjectionInKernel {
        axis_projection(
            space,
            comptime!(self.projection.clone()),
            self.map.clone(),
            comptime!(self.store.vector_size),
        )
    }

    /// The mutable twin of [`flat`](Memory::flat).
    pub(crate) fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("Tile::flat_mut: writing a packed tile requires repacking")
        }
        let extent = self.window.extent.clone();
        let check = self.write_check();
        FlatViewMut::new(
            self.window_view_mut::<W>(comptime!(Guard::Checked))
                .view_mut(FlatLayout::new(extent)),
            check,
        )
    }

    /// The `i`-th batch matrix as a writable 2-D view.
    pub(crate) fn matrix_mut<W: Size>(
        &mut self,
        i: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] space: Space,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        // A write aliases only where two logical positions share a cell, which is what an
        // overlapping map is; a partition is a bijection, so its windows tile and each cell is
        // written once. A gathered operand is read through `Tile::nd` and never written here.
        comptime!(assert!(
            self.projection.composition() != Composition::Overlapping,
            "Memory::matrix_mut: an overlapping operand aliases under a write"
        ));
        let layout = self.batch_matrix(space, axes, i);
        self.masked_mut::<W, Coords2d, ProjectedMatrix>(layout)
    }

    /// The [`AccumulateView`] over batch matrix `i`: [`matrix_mut`](Memory::matrix_mut) plus the
    /// [`UnitShare`] these cells carry, the [`Monoid`] they fold under and what the accumulation
    /// starts from, so a leaf accumulates through it without being told any of the three.
    pub(crate) fn matrix_accumulate<W: Size>(
        &mut self,
        i: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] space: Space,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W> {
        let unit_share = comptime!(self.unit_share);
        let split_share = comptime!(self.split_share);
        let write = comptime!(self.access.write);
        let init_from = comptime!(self.init_from);
        AccumulateView::new(
            self.matrix_mut::<W>(i, axes, space),
            unit_share,
            split_share,
            write,
            monoid,
            init_from,
        )
    }

    /// The [`AccumulateView`] over flat elements: [`flat_mut`](Memory::flat_mut) plus the
    /// [`UnitShare`] these cells carry and the [`Monoid`] they fold under.
    pub(crate) fn flat_accumulate<W: Size>(
        &mut self,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W, Coords1d> {
        // A flat logical scan only agrees with this physical window under the direct,
        // non-storage-tiled mapping. Otherwise the reduction's logical accumulator index would
        // seed and commit a different physical cell than the one it reduces for.
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "Memory::flat_accumulate: a storage-tiled window has no flat logical accumulator view"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "Memory::flat_accumulate: a gathered window has no flat logical accumulator view"
        ));
        let unit_share = comptime!(self.unit_share);
        let split_share = comptime!(self.split_share);
        let write = comptime!(self.access.write);
        let init_from = comptime!(self.init_from);
        AccumulateView::new(
            self.flat_mut::<W>(),
            unit_share,
            split_share,
            write,
            monoid,
            init_from,
        )
    }

    /// Window down to `region`: shift the origin by the region's tile coordinate times the
    /// sub-tile edge, crop each physical axis to the region it now covers, re-box the same buffer.
    /// `bound` is carried through unchanged, so the leaf masks correctly at any nesting depth.
    ///
    /// Under a gathering [`Projection`] a physical axis is an affine combination of axes, so its
    /// advance sums one term per contributing axis and its extent is the receptive field
    /// ([`Projection::span`]) rather than a single edge: consecutive sibling windows overlap.
    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> Memory<T> {
        let rank = comptime!(self.projection.physical_rank());
        let (origin, extent, advances, map) = if comptime!(self.projection.is_direct()) {
            self.direct_descent(step, comptime!(space.clone()))
        } else {
            self.gathered_descent(step)
        };
        let start = self
            .window_start
            .plus(advances.sum(comptime!((0..rank).collect::<Vec<_>>())));

        // Re-window the scales alongside the values.
        let mut origin_u32 = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            origin_u32.push(origin.at(p).cast::<u32>());
        }
        let quant = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => {
                comptime!(assert!(
                    !self.window.signed,
                    "Memory::at: a quantized operand cannot carry a negative window origin, its \
                     scale grid is addressed unsigned"
                ));
                // A quantized operand is direct (asserted at construction), so the child window's
                // extent per axis is this level's cut edge; an axis left whole keeps its extent.
                ComptimeOption::new_Some(info.window(
                    &origin_u32,
                    rank,
                    comptime!(self.store.vector_size),
                    comptime!(
                        (0..rank)
                            .map(|p| match step.level.extent_in(&space, space.axis_at(p)) {
                                Extent::Static(edge) => edge,
                                Extent::Dynamic => info.extent[p],
                            })
                            .collect::<Vec<_>>()
                    ),
                ))
            }
            ComptimeOption::None => ComptimeOption::new_None(),
        };

        self.moved_to(
            Window::new(
                origin,
                extent,
                self.window.bound.clone(),
                comptime!(self.window.signed),
                comptime!(self.window.boundaries.clone()),
            ),
            start,
            map,
            quant,
            // The window no longer covers the buffer, so the straight-through fill is off.
            comptime!(Access {
                whole: false,
                overhang: self.access.overhang,
                write: self.access.write,
                units: self.access.units,
                storage: storage_below(self.access.storage, step.depth, &step.level, &space),
            }),
            comptime!(UnitShare::new(&step.level, &space).under(self.unit_share)),
            // Joined level by level: the level's whole space still has the axis this operand's
            // projection dropped, which is what tells a split from a cut of the whole axis.
            comptime!(SplitShare::new(&step.level, &step.space, &space).under(self.split_share)),
            // The scales are windowed with the values they ride.
            self.factor.at(step),
        )
    }

    /// One level down under the direct mapping (one logical axis per physical axis at coefficient
    /// `1`): the child window's origin and extent per axis, each axis's line-route advance, and
    /// the integral map (nothing to carry, no phase left over).
    ///
    /// Its own loop because only this mapping can sit on a *tiled* buffer, where `step_offset`
    /// folds the grid/tile digit split a scaled advance cannot be pushed through.
    fn direct_descent(
        &self,
        step: &Step,
        #[comptime] space: Space,
    ) -> (Coords<i32>, Coords<u32>, Coords<u32>, RuntimeMap) {
        let mut origin = Coords::<i32>::new();
        let mut extent = Coords::<u32>::new();
        let mut advances = Coords::<u32>::new();
        let rank = comptime!(self.projection.physical_rank());
        let last = comptime!(rank - 1);
        let w = comptime!(self.store.vector_size);

        #[unroll]
        for p in 0..rank {
            let axis = space.axis_at(p);
            // An axis left whole over a dynamic extent has no edge to cut by: the window
            // carries through unmoved and uncropped.
            if comptime!(matches!(
                step.level.extent_in(&space, axis),
                Extent::Dynamic
            )) {
                origin.push(self.window.origin.at(p));
                extent.push(self.window.extent.at(p));
                advances.push(0u32);
            } else {
                // The innermost (vectorized) axis's edge is a line count, so `/ width`.
                let edge = comptime!(if p == last {
                    let e = step.level.extent_in(&space, axis).get();
                    // A padded stage's innermost extent need not fill whole lines, but the axis
                    // must be cut whole or the next region would begin mid-line. `extent_raw`: a
                    // `Dynamic` axis has no extent to be cut whole, so it owes the divisibility.
                    assert!(
                        e.is_multiple_of(w)
                            || matches!(space.extent_raw(axis), Extent::Static(x) if x == e),
                        "Memory::at: the innermost edge {e} is neither a whole number of \
                         {w}-wide lines nor the axis's whole extent ({:?}), so a step would \
                         start mid-line",
                        space.extent_raw(axis)
                    );
                    e.div_ceil(w)
                } else {
                    step.level.extent_in(&space, axis).get()
                });
                let index = step.coord(axis);

                origin.push(
                    self.window
                        .origin
                        .at(p)
                        .plus(index.times(edge).cast::<u32>().cast::<i32>()),
                );
                extent.push(comptime!(edge as u32).runtime());
                advances.push(index.cast::<u32>().times(step_offset(
                    comptime!(self.layout.projection.clone()),
                    comptime!(Axis(p as u8)),
                    edge,
                    &self.layout.physical_shape,
                    &self.layout.physical_strides,
                )));
            }
        }
        (origin, extent, advances, RuntimeMap::integral(rank))
    }

    /// [`direct_descent`](Memory::direct_descent) under a gathering mapping: each physical axis
    /// moves by the sum of its terms and covers the receptive field, and the phase each axis's
    /// division left over is this level's map; the coefficients are invariant down the descent.
    fn gathered_descent(&self, step: &Step) -> (Coords<i32>, Coords<u32>, Coords<u32>, RuntimeMap) {
        let mut origin = Coords::<i32>::new();
        let mut extent = Coords::<u32>::new();
        let mut advances = Coords::<u32>::new();
        let mut residues = Coords::<u32>::new();
        let rank = comptime!(self.projection.physical_rank());
        let w = comptime!(self.store.vector_size);

        #[unroll]
        for pa in 0..rank {
            let (moved, residue, span) =
                gathered_axis_descent(comptime!(self.projection.clone()), step, &self.map, w, pa);
            // The move only goes forward, so it adds directly to the signed origin.
            origin.push(self.window.origin.at(pa).plus(moved.cast::<i32>()));
            residues.push(residue);
            extent.push(span);
            // `Projection::validate` pins a gathered operand to untiled storage (bare gmem, or
            // the row-major compacted stage of one), so one physical axis step is one stride
            // and the advance passes straight through.
            advances.push(moved.times(self.layout.physical_strides.at(pa)));
        }
        let map = RuntimeMap {
            coefficients: self.map.coefficients.clone(),
            residues,
        };
        (origin, extent, advances, map)
    }

    /// This store looking at `window` from line `window_start` on, under `access`, `units` and
    /// `split_share`: the same buffer, layout, mapping and offsets, which no descent moves. What
    /// [`at`](Memory::at) and [`within`](Memory::within) build once they have settled the window.
    ///
    /// The layout addresses the whole buffer and never narrows; only the window moves. The mapping
    /// is the buffer's, invariant down the descent; the offsets only placed the top window, which
    /// the origin carries; a source window rides down as filled, a step moving both by one delta.
    #[allow(clippy::too_many_arguments)]
    fn moved_to(
        &self,
        window: Window,
        window_start: u32,
        map: RuntimeMap,
        quant: ComptimeOption<QuantInfo>,
        #[comptime] access: Access,
        #[comptime] unit_share: UnitShare,
        #[comptime] split_share: SplitShare,
        factor: Factor,
    ) -> Memory<T> {
        Memory::<T> {
            address: comptime!(self.address),
            store: Store::<T> {
                backing: self.store.backing.clone(),
                vector_size: comptime!(self.store.vector_size),
                quant,
                packing: comptime!(self.store.packing),
            },
            layout: self.layout.clone(),
            window,
            projection: comptime!(self.projection.clone()),
            source_window: self.source_window.clone(),
            lands: comptime!(self.lands),
            map,
            offsets: self.offsets.clone(),
            window_start,
            access,
            unit_share,
            split_share,
            init_from: comptime!(self.init_from),
            factor,
        }
    }

    /// This window placed at `from` on `axis` and reading no further than `until`, both counted
    /// in that axis's own elements.
    ///
    /// Where a routed coordinate names a whole tile of an axis, this places the window at an
    /// element and says where it stops (a packed sequence starts where the one before it ended).
    /// `until` arms [`Boundary::Zero`] on the axis, so an overrunning last tile's tail reads zero.
    pub(crate) fn within(&self, #[comptime] axis: Axis, from: usize, until: usize) -> Memory<T> {
        let proj = comptime!(self.projection.clone());
        comptime!(assert!(
            proj.untiled().is_direct() && !proj.is_tiled(),
            "Memory::within: placing a window at an element needs a direct, untiled mapping"
        ));
        let rank = comptime!(proj.physical_rank());
        let at = comptime!(proj.position(axis));

        let mut origin = Coords::<i32>::new();
        let mut bound = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            if comptime!(p == at) {
                origin.push(self.window.origin.at(p).plus(from.cast::<i32>()));
                bound.push(self.window.bound.at(p).min_with(until.cast::<u32>()));
            } else {
                origin.push(self.window.origin.at(p));
                bound.push(self.window.bound.at(p));
            }
        }

        // The line route, which `dense_lines` and the matrix view read, moves by the same
        // elements: one axis step at edge `1`.
        let start = self.window_start.plus(from.cast::<u32>().times(step_offset(
            comptime!(self.layout.projection.clone()),
            comptime!(Axis(at as u8)),
            1usize,
            &self.layout.physical_shape,
            &self.layout.physical_strides,
        )));

        self.moved_to(
            Window::new(
                origin,
                self.window.extent.clone(),
                bound,
                comptime!(self.window.signed),
                comptime!({
                    let mut boundaries = self.window.boundaries.clone();
                    if boundaries.is_empty() {
                        boundaries = (0..rank).map(|_| None).collect();
                    }
                    boundaries[at] = Some(Boundary::Zero);
                    boundaries
                }),
            ),
            start,
            self.map.clone(),
            self.store.quant.clone(),
            // A placed window no longer covers the buffer, so the straight-through fill is off,
            // and `until` is a runtime bound the launch could not have stated: reads past it are
            // an overhang this tile did not have before, so it masks from here down.
            comptime!(Access {
                whole: false,
                overhang: Overhang::Masked,
                write: self.access.write,
                units: self.access.units,
                storage: self.access.storage,
            }),
            comptime!(self.unit_share),
            comptime!(self.split_share),
            // Placing a window moves the values, and the scales ride them unchanged.
            self.factor.clone(),
        )
    }
}

/// What the storage tiles are to the window one level down: descending through the storage tile's
/// own level puts the window inside one storage tile, where it stays. The launch matched the tile
/// to that level; here it only has to hand every axis down static and never be skipped past.
fn storage_below(storage: Storage, depth: usize, level: &Level, space: &Space) -> Storage {
    match storage {
        Storage::Strided => Storage::Strided,
        Storage::Contiguous => Storage::Contiguous,
        Storage::Tiled(None) => Storage::Tiled(None),
        Storage::Tiled(Some(tiled_at)) => {
            assert!(
                depth <= tiled_at,
                "Memory::at: this window is above its storage tile, the tile of level \
                 {tiled_at}, yet the descent is at depth {depth}; the storage tile's level was \
                 skipped past"
            );
            if depth < tiled_at {
                return Storage::Tiled(Some(tiled_at));
            }
            for axis in space.axes() {
                assert!(
                    matches!(level.extent_in(space, axis), Extent::Static(_)),
                    "Memory::at: level {tiled_at} hands {axis:?} down dynamic, so its tile is \
                     no storage tile"
                );
            }
            Storage::Contiguous
        }
    }
}

/// One gathered physical axis's descent into `region`: how far its window moves, the phase that
/// move leaves behind, and the receptive field the child then covers.
///
/// The move sums one term per contributing axis (tile coordinate × sub-tile edge × coefficient),
/// all comptime but the coordinate, so it stays the multiply-add `window_start` promises.
///
/// Each term divides by `vector_size` on its own, which only sums back to the whole move because
/// the innermost axis carries a single identity term, as [`Projection::validate`] requires;
/// a second term there would need the division after the sum, not before.
///
/// A [rational](crate::Divisor) axis moves by the whole cells its numerator crossed and hands the
/// phase it did not fill to the child: `⌊(move + phase)/d⌋` splits into this step plus a child
/// floor starting at the new phase, which is what makes the descent compose across levels.
#[cube]
fn gathered_axis_descent(
    #[comptime] projection: Projection,
    cut: &Step,
    map: &RuntimeMap,
    #[comptime] vector_size: usize,
    #[comptime] pa: usize,
) -> (u32, u32, u32) {
    let axis_map = comptime!(projection.physical_axis(pa));
    let n = comptime!(axis_map.terms().len());
    let picks = comptime!((0..n).collect::<Vec<_>>());
    let lined = comptime!(pa == projection.physical_rank() - 1);

    let mut terms = Coords::<u32>::new();
    // One receptive-field term per contributing axis, `(edge - 1) * scale`. The field's leading
    // `1` is the branch's to add: under a division it is the quotient that carries it, not the
    // numerator.
    let mut spans = Coords::<u32>::new();
    #[unroll]
    for t in 0..n {
        let term = comptime!(axis_map.terms()[t]);
        let edge = comptime!(cut.level.extent_in(&cut.space, term.axis).get());
        match comptime!(term.scale) {
            Scale::Static(s) => {
                let step = comptime!(if lined {
                    // A window along the lined axis starts at a whole line, or it is the whole
                    // axis and starts at its origin; any other cut would place its origin
                    // inside a line, which a line index cannot say.
                    assert!(
                        (edge * s).is_multiple_of(vector_size)
                            || matches!(cut.space.extent_raw(term.axis), Extent::Static(x) if x == edge),
                        "Memory::at: the innermost edge {edge} of {:?} is neither a whole number \
                         of {vector_size}-wide lines nor the axis's whole extent, so a step would \
                         start mid-line",
                        term.axis
                    );
                    edge * s / vector_size
                } else {
                    edge * s
                });
                terms.push(cut.coord(term.axis).times(step).cast::<u32>());
                spans.push(comptime!(((edge - 1) * s) as u32).runtime());
            }
            // The line division above never meets a runtime coefficient: the innermost physical
            // axis is a single identity term, which `Projection::validate` requires and `Static`
            // is the only spelling of.
            Scale::Dynamic { .. } => {
                let coefficient = map
                    .coefficients
                    .at(comptime!(projection.dynamic_scale_index(pa, t).unwrap()));
                terms.push(
                    cut.coord(term.axis)
                        .cast::<u32>()
                        .times(comptime!(edge as u32).runtime())
                        .times(coefficient),
                );
                spans.push(comptime!((edge - 1) as u32).runtime().times(coefficient));
            }
        }
    }
    let advance = terms.sum(comptime!(picks.clone()));

    if comptime!(!axis_map.is_rational()) {
        // The receptive field of the child edges: `1 + Σ (edge - 1) * scale`, which stays comptime
        // for the mapping that is.
        let span = if comptime!(!axis_map.has_dynamic_scale()) {
            comptime!({
                let s = projection.span(pa, |a| cut.level.extent_in(&cut.space, a).get());
                (if lined { s / vector_size } else { s }) as u32
            })
            .runtime()
        } else {
            spans.sum(comptime!(picks.clone())).plus(1)
        };
        (advance, 0u32, span)
    } else {
        // No `/ vector_size` anywhere below, and none is owed: `Projection::validate` refuses a
        // rational innermost physical axis at any width past `1`, so it is `1` whenever this
        // branch runs and the terms above are already in elements.
        let numerator = advance.plus(map.residues.at(pa));
        let field = spans.sum(comptime!(picks.clone()));
        match comptime!(axis_map.divisor()) {
            Divisor::Static(d) => {
                let d = comptime!(d as u32);
                let residue = numerator.remainder(d);
                (
                    numerator.divided_by(d),
                    residue,
                    field.plus(residue).divided_by(d).plus(1),
                )
            }
            Divisor::Dynamic { .. } => {
                let d = map
                    .coefficients
                    .at(comptime!(projection.dynamic_divisor_index(pa).unwrap()));
                let residue = numerator.remainder(d);
                (
                    numerator.divided_by(d),
                    residue,
                    field.plus(residue).divided_by(d).plus(1),
                )
            }
        }
    }
}
