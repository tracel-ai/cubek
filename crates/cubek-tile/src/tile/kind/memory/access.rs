//! Reading, writing and windowing a [`Memory`]: cooperative fills, the views a leaf reads and
//! writes through, and [`at`](Memory::at).
//! Cooperative fills assume every unit their destination names ([`FillUnits`]) runs them: the
//! cube's, or, for a stage one plane owns, that plane's.

use cubecl::{
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, ErasedTensor, View, ViewMut, WriteOnly,
        layout::{Coordinates, Coords1d, Coords2d, CoordsDyn},
    },
};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// A read [`View`] over `Vector<T, W>` lines through the base layout and `Window`.
    /// `W` is the line width (`self.store.vector_size`).
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

    /// Scalars of this stage one unit holds in registers across a contraction.
    #[allow(dead_code)] // Reached through its expand, from `pipelined_through_registers`.
    pub(crate) fn fetched_scalars(&self) -> comptime_type!(usize) {
        self.mem("fetched_scalars").fetched_scalars()
    }

    /// Free the shared memory this stage is held in; a stage of a call has none.
    pub(crate) fn free_stage(&self) {
        self.mem("free_stage").store.free();
    }

    /// This unit's share of filling this stage from `src`, read into `fetched` but not yet written.
    #[allow(dead_code)] // Reached through its expand, from `pipelined_through_registers`.
    pub(crate) fn fetch_from<W: Size>(&self, src: &Tile<T>, fetched: &mut Array<Vector<T, W>>) {
        let space = comptime!(self.place.space.clone());
        self.mem("fetch_from")
            .fetch_straight(src.mem("fetch_from"), space, fetched);
    }

    /// Write what [`fetch_from`](Tile::fetch_from) read into this stage.
    #[allow(dead_code)] // Reached through its expand, from `pipelined_through_registers`.
    pub(crate) fn store_fetched<W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        self.mem_mut("store_fetched").store_fetched(fetched);
    }

    pub fn view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        let g = self.mem_mut("view_mut");
        g.window_view_mut::<W>(comptime!(Guard::Checked))
    }
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// State what the accumulation being lowered starts from.
    pub(crate) fn set_init_from(&mut self, #[comptime] init_from: InitFrom) {
        comptime!({
            self.init_from = init_from;
        });
    }

    /// State how a contraction into this window runs.
    pub(crate) fn set_contraction(&mut self, #[comptime] contraction: Contraction) {
        comptime!({
            self.contraction = Some(contraction);
        });
    }

    /// Zero this window; a checked window skips cells past the logical bound.
    pub(crate) fn zero(&mut self) {
        self.init(T::from_int(0));
    }

    /// Initialize this window with `val`; a checked window skips cells past the logical bound.
    pub(crate) fn init(&mut self, val: T) {
        let size!(W) = comptime!(self.store.vector_size);
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        for i in 0..total {
            d.write(i, Vector::<T, W>::cast_from(val));
        }
    }

    /// What a stage of this store holds: served values if plain, stored words if packed.
    pub(crate) fn stage_element(&self) -> comptime_type!(StageElement) {
        comptime!(match self.store.packing {
            Packing::Plain => StageElement::Served,
            Packing::Packed { .. } => StageElement::Stored,
        })
    }

    /// How this store's values sit in memory, as stated at construction.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(self.store.packing)
    }

    /// This buffer's byte length: the transaction count a TMA fill into it lands.
    pub(crate) fn size_bytes(&self) -> u32 {
        let lines = self.store.buffer().len() as u32;
        let wp = comptime!(self.store.packing.physical(self.store.vector_size) as u32);
        match comptime!(self.store.packing) {
            Packing::Plain => lines * T::size().comptime() as u32 * wp,
            Packing::Packed { field: _ } => lines * u32::size().comptime() as u32 * wp,
        }
    }

    /// The base layout: a `[grid…, tile…]` split (`levels > 0`) or plain strided (`levels = 0`).
    pub(crate) fn base(&self) -> BufferLayout {
        self.layout.clone()
    }

    fn window(&self) -> Window {
        self.window.clone()
    }

    /// The window extent, for shape-only readers that must not regroup the buffer.
    fn extent(&self) -> Coords<u32> {
        self.window.extent.clone()
    }

    /// The buffer re-grouped into `Vector<T, W>` lines; buffers only.
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.store.buffer().as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](Memory::lines); buffers only.
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.store
            .buffer_mut()
            .as_vectorized_mut()
            .with_vector_size_mut::<W>()
    }

    /// The backing as a [`ViewMut`] addressed by `layout`: the write path.
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

    /// The backing as a [`View`] addressed by `layout`: the read path.
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

    /// The window as a view over its own coordinates, served at `T` grouped `W` wide.
    fn window_view<W: Size>(&self, #[comptime] guard: Guard) -> View<'_, Vector<T, W>, CoordsDyn> {
        match comptime!(self.access.storage) {
            WindowStorage::Contiguous => {
                let start = self.window_start.cast::<usize>();
                let all = self.lines::<W>();
                all.slice(start, all.len()).view(self.contiguous_layout())
            }
            WindowStorage::Stored(_) => self
                .read_view::<W>(self.base())
                .view(self.window().with_guard(guard)),
        }
    }

    /// [`window_view`](Memory::window_view) at the storage element `I`, grouped `WP` wide.
    pub(crate) fn window_view_storage<I: Numeric, WP: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> View<'_, Vector<I, WP>, CoordsDyn> {
        match comptime!(self.access.storage) {
            WindowStorage::Contiguous => {
                let start = self.window_start.cast::<usize>();
                let all = self.lines_storage::<I, WP>();
                all.slice(start, all.len()).view(self.contiguous_layout())
            }
            WindowStorage::Stored(_) => self
                .lines_storage::<I, WP>()
                .view(self.base())
                .view(self.window().with_guard(guard)),
        }
    }

    /// The mutable twin of [`window_view`](Memory::window_view).
    fn window_view_mut<W: Size>(
        &mut self,
        #[comptime] guard: Guard,
    ) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match comptime!(self.access.storage) {
            WindowStorage::Contiguous => {
                let start = self.window_start.cast::<usize>();
                let layout = self.contiguous_layout();
                let all = self.lines_mut::<W>();
                let len = all.len();
                all.slice_mut(start, len).view_mut(layout)
            }
            WindowStorage::Stored(_) => {
                let base = self.base();
                let window = self.window().with_guard(guard);
                self.write_view::<W>(base).view_mut(window)
            }
        }
    }

    /// The layout of a window inside one storage tile, relative to its origin.
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
            rows: RowArrangement::InOrder,
        }
    }

    /// [`lines`](Memory::lines) re-typed to the storage element `I` (`u32` words when packed).
    fn lines_storage<I: Numeric, W: Size>(&self) -> &[Vector<I, W>] {
        let storage = unsafe { self.store.buffer().downcast_unchecked::<I>() };
        storage.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines_storage`](Memory::lines_storage).
    pub(crate) fn lines_storage_mut<I: Numeric, W: Size>(&mut self) -> &mut [Vector<I, W>] {
        let storage = unsafe { self.store.buffer_mut().downcast_mut_unchecked::<I>() };
        storage.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// The buffer from the window origin on, rows stepping by [`row_stride`](Memory::row_stride).
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

    /// Line offset of the window origin; the window must be one contiguous region.
    fn window_offset(&self) -> usize {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "Memory::window_offset: cmma cannot mask an overhang"
        ));
        // A base and row stride above a storage tile would silently read another tile's cells.
        match comptime!(self.access.storage) {
            WindowStorage::Stored(Storage::Strided) => {}
            WindowStorage::Contiguous => {}
            WindowStorage::Stored(Storage::Tiled(Some(level))) => panic!(
                "Memory::window_offset: this window sits above its storage tile (the tile of \
                 level {level}), spanning several, so it is not one contiguous region; descend \
                 through that level first, or read the operand through its layout"
            ),
            WindowStorage::Stored(Storage::Tiled(None)) => panic!(
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
        // A raw window would hand a packed store's words over as served values.
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "Memory::window_slice: a packed store has no raw element window; a fragment \
                 load reads it through Tile::matrix_packed"
            )
        }
        self.window_start.cast::<usize>()
    }

    /// Scalar stride between matrix rows.
    pub(crate) fn row_stride(&self) -> u32 {
        let rank = comptime!(self.layout.projection.physical_rank());
        self.row_stride_at(comptime!(rank - 2))
    }

    /// [`row_stride`](Memory::row_stride) with the row axis stated (direct, untiled stores only).
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

    /// Re-view this buffer through `layout` as a [`Masked`] carrying its own mask flag.
    pub(crate) fn masked<W: Size, C: Coordinates, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "Tile::matrix: a packed tile only serves values its read unpacks \
                 (Tile::matrix_packed)"
            )
        }
        Masked::new(
            self.window_view::<W>(guard).view(layout),
            comptime!(guard.checks() && self.access.overhang.masks()),
        )
    }

    /// The mask flag for a write view; refuses a [`Boundary::Clamp`] operand, whose writes alias.
    fn write_check(&self) -> comptime_type!(bool) {
        // Whole-operand on purpose: one clamped axis already folds two cells onto one.
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

    /// Re-view this buffer as a flat 1-D [`FlatView`] over its [`Window`] extent.
    pub(crate) fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        FlatView::new(
            self.window_view::<W>(comptime!(Guard::Checked))
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// [`flat`](Memory::flat) unpacked through [`Packing`]; `WP` is the physical line.
    pub(crate) fn flat_unpacked<WP: Size, W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        self.unpacked::<WP, W, Coords1d, FlatLayout>(
            FlatLayout::new(self.window.extent.clone()),
            comptime!(Guard::Checked),
        )
    }

    /// [`masked`](Memory::masked) unpacked through [`Packing`].
    /// `WP` is the physical line, `W` the served one.
    pub(crate) fn unpacked<WP: Size, W: Size, C: Coordinates + 'static, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        let packing = self.packing();
        match comptime!(packing) {
            Packing::Plain => self.masked::<W, C, L>(layout, guard),
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

    /// [`unpacked`](Memory::unpacked) with the physical line taken from this store's [`Packing`].
    pub(crate) fn packed<W: Size, C: Coordinates + 'static, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, C> {
        let packing = self.packing();
        let physical = comptime!(packing.physical(self.store.vector_size));
        // `size!` binds inside each arm: hoisted, a packed read lands on the wrong field.
        match comptime!(packing) {
            Packing::Plain => {
                let size!(WP) = physical;
                self.unpacked::<WP, W, C, L>(layout, guard)
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = physical;
                self.unpacked::<WP, W, C, L>(layout, guard)
            }
        }
    }

    /// The raw words of a packed store over the tile's whole logical box.
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

    /// The identity step over this window's physical box, for callers that fold the map themselves.
    /// It masks against the physical box only: the caller owes in-range logical coordinates.
    pub(crate) fn physical_box(&self) -> CompactionStep {
        let rank = comptime!(self.projection.physical_rank());
        CompactionStep::new(self.window.extent.clone(), comptime!(vec![1; rank]))
    }

    /// The `i`-th batch matrix of this window over `axes`, through the operand's mapping.
    pub(crate) fn batch_matrix(
        &self,
        #[comptime] space: Space,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> ProjectedMatrix {
        let bound = self.extent();
        let load = self.vector_tile(&space);
        ProjectedMatrix::new(
            TileMatrix::batch(
                &bound,
                comptime!(&space),
                comptime!(self.projection.composition()),
                comptime!(&load),
                axes,
                i,
            ),
            self.axis_projection(comptime!(space.clone())),
        )
    }

    /// This window's whole logical box as a `rows x cols` matrix, through the operand's mapping.
    pub(crate) fn whole_matrix(
        &self,
        #[comptime] space: Space,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> ProjectedMatrix {
        let vector_size = comptime!(self.store.vector_size);
        ProjectedMatrix::new(
            TileMatrix::whole(comptime!(&space), vector_size, rows, cols),
            axis_projection(
                comptime!(space),
                comptime!(self.projection.clone()),
                self.map.clone(),
                vector_size,
            ),
        )
    }

    /// The operand's [`Projection`] applied to this window's logical box: the N-D read surface,
    /// each axis counted in this memory's loads ([`vector_tile`](Memory::vector_tile)).
    pub(crate) fn axis_projection(&self, #[comptime] space: Space) -> ProjectionInKernel {
        let load = self.vector_tile(&space);
        ProjectionInKernel::new(
            Coords::constant(comptime!(load.counts(&space))),
            self.map.clone(),
            comptime!(space.clone()),
            comptime!(self.projection.clone()),
            comptime!(load.values()),
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
        // Only an overlapping map aliases a cell under a write.
        comptime!(assert!(
            self.projection.composition() != Composition::Overlapping,
            "Memory::matrix_mut: an overlapping operand aliases under a write"
        ));
        let layout = self.batch_matrix(space, axes, i);
        self.masked_mut::<W, Coords2d, ProjectedMatrix>(layout)
    }

    /// The [`AccumulateView`] over batch matrix `i`, carrying its share, [`Monoid`] and init.
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

    /// The [`AccumulateView`] over flat elements.
    pub(crate) fn flat_accumulate<W: Size>(
        &mut self,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W, Coords1d> {
        // Only under a direct, untiled mapping is the flat logical index the physical cell.
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

    /// Window down to the region `step` names, re-boxing the same buffer; `bound` carries through.
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
            // No longer covers the buffer, so no straight-through fill.
            comptime!(Access {
                whole: false,
                overhang: self.access.overhang,
                write: self.access.write,
                fill: self.access.fill,
                storage: storage_below(self.access.storage, step.depth, &step.level, &space),
                delivery: self.access.delivery,
            }),
            comptime!(UnitShare::new(&step.level, &space).under(self.unit_share)),
            // Per level: the level's space still has the axis the projection dropped.
            comptime!(SplitShare::new(&step.level, &step.space, &space).under(self.split_share)),
            self.factor.at(step),
        )
    }

    /// One level down under the direct mapping: per-axis origin, extent, advance, and integral map.
    fn direct_descent(
        &self,
        step: &Step,
        #[comptime] space: Space,
    ) -> (Coords<i32>, Coords<u32>, Coords<u32>, RuntimeMap) {
        let mut origin = Coords::<i32>::new();
        let mut extent = Coords::<u32>::new();
        let mut advances = Coords::<u32>::new();
        let rank = comptime!(self.projection.physical_rank());
        let load = self.vector_tile(&space);

        #[unroll]
        for p in 0..rank {
            let axis = space.axis_at(p);
            // A whole axis over a dynamic extent carries through unmoved.
            if comptime!(matches!(
                step.level.extent_in(&space, axis),
                Extent::Dynamic
            )) {
                origin.push(self.window.origin.at(p));
                extent.push(self.window.extent.at(p));
                advances.push(0u32);
            } else {
                // The edge counts loads: along an axis a load spans, over its extent there.
                let edge = comptime!(load.loads_in(
                    axis,
                    step.level.extent_in(&space, axis).get(),
                    space.extent_raw(axis)
                ));
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

    /// [`direct_descent`](Memory::direct_descent) under a gathering mapping.
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
            origin.push(self.window.origin.at(pa).plus(moved.cast::<i32>()));
            residues.push(residue);
            extent.push(span);
            // `Projection::validate` keeps a gathered operand untiled: one axis step is one stride.
            advances.push(moved.times(self.layout.physical_strides.at(pa)));
        }
        let map = RuntimeMap {
            coefficients: self.map.coefficients.clone(),
            residues,
        };
        (origin, extent, advances, map)
    }

    /// This store looking at `window` from line `window_start` on; buffer and layout unchanged.
    #[allow(clippy::too_many_arguments)]
    fn moved_to(
        &self,
        window: Window,
        window_start: u32,
        map: RuntimeMap,
        #[comptime] access: Access,
        #[comptime] unit_share: UnitShare,
        #[comptime] split_share: SplitShare,
        factor: Factor,
    ) -> Memory<T> {
        Memory::<T> {
            address: comptime!(self.address),
            store: self.store.clone(),
            layout: self.layout.clone(),
            window,
            projection: comptime!(self.projection.clone()),
            source_window: self.source_window.clone(),
            map,
            offsets: self.offsets.clone(),
            window_start,
            access,
            unit_share,
            split_share,
            init_from: comptime!(self.init_from),
            contraction: comptime!(self.contraction),
            factor,
            codebook: self.codebook.clone(),
        }
    }

    /// This window filled by the units of `holder` where they are fewer than the ones that fill
    /// its buffer: a plane's window of a buffer the cube fills is the plane's to fill.
    pub(crate) fn filled_by(&self, #[comptime] holder: ComputeScope) -> Memory<T> {
        let fill = comptime!(match holder < self.access.fill.scope {
            true => FillUnits {
                scope: holder,
                count: 0,
            },
            false => self.access.fill,
        });
        self.moved_to(
            self.window.clone(),
            self.window_start,
            self.map.clone(),
            comptime!(Access {
                fill,
                ..self.access
            }),
            comptime!(self.unit_share),
            comptime!(self.split_share),
            self.factor.clone(),
        )
    }

    /// This window placed at element `from` on `axis`, reading up to `until` (zero past it).
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
            // `until` is a runtime bound, so the window masks from here down.
            comptime!(Access {
                whole: false,
                overhang: Overhang::Masked,
                write: self.access.write,
                fill: self.access.fill,
                storage: self.access.storage,
                delivery: self.access.delivery,
            }),
            comptime!(self.unit_share),
            comptime!(self.split_share),
            self.factor.clone(),
        )
    }
}

/// The storage tiling one level down: through the storage tile's own level, inside one tile.
fn storage_below(
    storage: WindowStorage,
    depth: usize,
    level: &Level,
    space: &Space,
) -> WindowStorage {
    match storage {
        WindowStorage::Contiguous
        | WindowStorage::Stored(Storage::Strided | Storage::Tiled(None)) => storage,
        WindowStorage::Stored(Storage::Tiled(Some(tiled_at))) => {
            assert!(
                depth <= tiled_at,
                "Memory::at: this window is above its storage tile, the tile of level \
                 {tiled_at}, yet the descent is at depth {depth}; the storage tile's level was \
                 skipped past"
            );
            if depth < tiled_at {
                return storage;
            }
            for axis in space.axes() {
                assert!(
                    matches!(level.extent_in(space, axis), Extent::Static(_)),
                    "Memory::at: level {tiled_at} hands {axis:?} down dynamic, so its tile is \
                     no storage tile"
                );
            }
            WindowStorage::Contiguous
        }
    }
}

/// One gathered physical axis's descent: its window move, leftover phase and receptive field.
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
    // Receptive-field terms `(edge - 1) * scale`; each branch adds the leading `1`.
    let mut spans = Coords::<u32>::new();
    #[unroll]
    for t in 0..n {
        let term = comptime!(axis_map.terms()[t]);
        let edge = comptime!(cut.level.extent_in(&cut.space, term.axis).get());
        match comptime!(term.scale) {
            Scale::Static(s) => {
                let step = comptime!(if lined {
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
            // No line division: `Projection::validate` keeps the innermost axis a static identity.
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
        // No `/ vector_size`: `Projection::validate` forces width `1` on a rational innermost axis.
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
