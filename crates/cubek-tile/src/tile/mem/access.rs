//! Touching a [`MemData`]: filling one from another (the cooperative copy), the views a leaf
//! reads and writes it through, and [`at`](MemData::at), which windows it down to a region.

use cubecl::{
    prelude::*,
    quant::scheme::{QuantStore, QuantValue},
    std::quant::unpack_fields,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, ErasedTensor, View, ViewMut, WriteOnly,
        layout::{Coordinates, Coords1d, Coords2d, CoordsDyn},
    },
};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// A read [`View`] over `Vector<T, W>` lines: the scalar buffer re-grouped into its physical
    /// width, then re-viewed through the base layout and [`Window`]. `W` is the line width
    /// (`self.store.vector_size`); pass `Const<1>` when only the (width-invariant) leading shape is needed.
    pub fn view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                if comptime!(g.store.packing != Packing::Plain) {
                    panic!(
                        "Tile::view: a packed tile only serves values its read unpacks \
                         (Tile::copy_from, Tile::matrix_transparent)"
                    )
                }
                let base = g.base();
                g.read_view::<W>(base).view(g.window())
            }
            TileKind::TmaGmem(_) => panic!("Tile::view: a tma source has no element view"),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::view: a plane tile has no memory view")
            }
            TileKind::Procedural(_) => panic!("Tile::view: a procedural tile has no memory view"),
        }
    }

    pub fn view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                if comptime!(g.store.quant.is_some()) {
                    panic!("Tile::view_mut: writing a quantized tile requires requantization")
                }
                let base = g.base();
                let window = g.window();
                g.write_view::<W>(base).view_mut(window)
            }
            TileKind::TmaGmem(_) => panic!("Tile::view_mut: a tma source has no element view"),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::view_mut: a plane tile has no memory view")
            }
            TileKind::Procedural(_) => panic!("Tile::view_mut: a procedural tile is not writable"),
        }
    }
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// State what the accumulation being lowered starts from ([`init`](MemData::init)).
    pub(crate) fn set_init_from(&mut self, #[comptime] init_from: InitFrom) {
        comptime!({
            self.init_from = init_from;
        });
    }

    /// Memory transport leaf: cooperative cyclic copy of `src` into `self`, whole
    /// `Vector<T, W>` lines at `self`'s width, unit `u` moving lines `u`, `u + CUBE_DIM`, ….
    /// The caller owns the rendezvous: a `sync_cube` must separate this fill from its readers.
    ///
    /// `space` is the logical space both sides carry. A gathered `src` stages the compacted
    /// *window* rather than its logical tile, so the fill stays a box copy and the gather stays at
    /// the leaf's read; see [`fill_straight`](MemData::fill_straight) and [`Compaction`].
    pub(crate) fn fill_from(&mut self, src: &MemData<T>, #[comptime] space: Space) {
        let size!(W) = comptime!(self.store.vector_size);
        let gathered = comptime!(!src.projection.is_direct());
        // A gathered stage keeps where it read from: the copy below writes the boundary's value
        // for every tap outside `src`, and nothing in the staged window says which those were.
        #[comptime]
        match &mut self.source_window {
            ComptimeOption::Some(source) => {
                source.origin = src.window.origin.clone();
                source.bound = src.window.bound.clone();
            }
            ComptimeOption::None => {}
        }
        if comptime!(self.store.quant.is_some()) {
            // Unreachable in practice: `Tile::of_impl` already asserts `quant.is_none() ||
            // coords.is_direct()` at construction, so a gathered `src` never carries a quantized
            // form to begin with. Kept as defense in case that invariant ever loosens.
            comptime!(assert!(
                !gathered,
                "MemData::fill_from: a gathered operand cannot stage in its quantized form"
            ));
            // Quant → quant: stage the packed storage words verbatim through the straight-line fill,
            // then the scales beside them, so a leaf read dequantizes straight out of smem with no
            // f32 inflation. A quantized stage is always a fresh whole buffer, so the masked slow
            // path never applies.
            comptime!(assert!(
                self.access.whole && !self.access.overhang.masks(),
                "MemData::fill_from: a quantized stage is always a fresh whole buffer"
            ));
            #[comptime]
            match &src.store.quant {
                ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                    // Unpacked: one element per value, so the physical line is the served line.
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => {
                            self.fill_straight::<i8, W>(src, comptime!(space.clone()))
                        }
                        other => panic!(
                            "MemData::fill_from: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    // Packed: the buffer holds `u32`s carrying `num_quants` values each, so the
                    // physical line is that much narrower than the served one.
                    QuantStore::PackedU32(_) => {
                        let size!(WP) =
                            comptime!(self.store.vector_size / info.scheme.num_quants());
                        self.fill_straight::<u32, WP>(src, comptime!(space.clone()));
                    }
                    other => panic!(
                        "MemData::fill_from: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                },
                ComptimeOption::None => panic!(
                    "MemData::fill_from: a quantized stage must be filled from a quantized source"
                ),
            }
            self.stage_scales(src);
        } else if comptime!(
            self.access.whole
                && !self.access.overhang.masks()
                && src.store.packing == Packing::Plain
        ) {
            // Plain → plain, whole destination: fill in destination-physical order (the write is
            // linear and only the source decodes, once per line by constants on a static store).
            // A padded stage is served in lines its source cannot hand out whole, so assemble each
            // destination line lane by lane.
            if comptime!(self.store.vector_size != src.store.vector_size) {
                comptime!(assert!(
                    src.store.vector_size == 1,
                    "MemData::fill_from: a padded stage is filled from a plain scalar operand, \
                     but its source serves {}-wide lines",
                    src.store.vector_size
                ));
            }
            self.fill_straight::<T, W>(src, comptime!(space.clone()));
        } else {
            // The general path reads the source as a flat run of its *window* and writes the
            // destination as a flat run of its own, which pairs the two only when they are the same
            // box. A gathered side's window is a physical box its logical rank does not match, so it
            // is addressed per axis or not at all. Reached by a windowed or masked destination, and
            // by a quantized source serving a plain one.
            comptime!(assert!(
                !gathered && self.projection.is_direct(),
                "MemData::fill_from: a gathered tile fills only a whole, unmasked, unquantized \
                 destination (a stage)"
            ));
            // The read decodes at the source's true storage element: `T` for a plain tile, else the
            // quantized store's element recovered from its scheme (the tile serves `T`, so `I` was
            // erased at construction and lives only on the scheme). This is what lets a plain
            // `copy_from`/`fill` dequantize on its own into a plain destination; the kernel never
            // threads `I`.
            #[comptime]
            match &src.store.quant {
                ComptimeOption::None => {
                    comptime!(assert!(
                        self.store.vector_size == src.store.vector_size,
                        "MemData::fill_from: a plain source is scanned at the destination's width, \
                         so a padded stage has to take the straight fill"
                    ));
                    // Equal widths here, so this only asks that the innermost extent is whole
                    // lines: `storage_extents` rounds it up, and nothing on the scan path would
                    // otherwise notice the last line the source cannot fill.
                    comptime!(fill_extent(
                        &space,
                        src.store.vector_size,
                        self.store.vector_size,
                        src.access.overhang.masks()
                    ));
                    // No scales to fold in, so the source's own packing is the whole read: served
                    // as stored, or unpacked from the words it holds into this plain destination.
                    let packing = src.packing();
                    match comptime!(packing) {
                        Packing::Plain => self.scan_transparent::<T, W, W>(src),
                        Packing::Native => panic!(
                            "MemData::fill_from: a native store with nothing to fold in serves \
                             its own element; bind it as that element"
                        ),
                        Packing::Packed { field: _ } => {
                            let size!(WP) = comptime!(packing.physical(src.store.vector_size));
                            self.scan_transparent::<u32, WP, W>(src)
                        }
                    }
                }
                ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => self.scan_transparent::<i8, W, W>(src),
                        other => panic!(
                            "MemData::fill_from: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    QuantStore::PackedU32(_) => {
                        if comptime!(src.store.vector_size == self.store.vector_size) {
                            let size!(WP) =
                                comptime!(src.store.vector_size / info.scheme.num_quants());
                            self.scan_transparent::<u32, WP, W>(src)
                        } else {
                            // The source's line is one whole word and this stage
                            // is narrower: unpack each word across several lines.
                            self.scan_words::<W>(src)
                        }
                    }
                    other => panic!(
                        "MemData::fill_from: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                },
            }
        }
    }

    /// The straight-line half of [`fill_from`](MemData::fill_from): the destination filled in its
    /// own physical order, whole `Vector<I2, WP2>` lines, decoding the source once per line rather
    /// than per cell. `I2` / `WP2` are the *storage* element and physical width: served for a plain
    /// copy, packed `u32` (or native `i8`) for a quant stage.
    ///
    /// Both sides are physical boxes of the same rank here, so the fill copies and never gathers;
    /// a gathered pair differs only by the compaction's step. The [`Window`] sits below either way,
    /// so a cell past the source's bound masks to zero once, at fill, not at every read.
    fn fill_straight<I2: Numeric, WP2: Size>(
        &mut self,
        src: &MemData<T>,
        #[comptime] space: Space,
    ) {
        // A gathered stage owns mutable map registers alongside its bytes. Store the source
        // window's coefficients and phase into those registers so bytes and interpretation are
        // one slot value. Direct stages carry no runtime map state.
        if comptime!(self.projection.is_rational() || self.projection.has_dynamic_scales()) {
            self.map.store_from(&src.map);
        }
        let check = comptime!(src.access.overhang.masks());
        let sw = comptime!(src.store.vector_size);
        let w = comptime!(self.store.vector_size);
        let compaction = comptime!(stage_compaction(
            &src.projection,
            &self.projection,
            w,
            &space
        ));
        // Empty exactly when the window has no holes to skip, so the fill reads the source box
        // straight through and this layer is never built.
        let steps = comptime!(match &compaction {
            Some(c) if !c.is_dense() => c.steps().to_vec(),
            _ => Vec::new(),
        });
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        let total = shape
            .fproduct(comptime!((0..plen).collect::<Vec<_>>()))
            .fcast::<usize>();
        let projection = comptime!(self.layout.projection.clone());
        // Asked whatever the widths: an equal-width fill reads nothing off the extent, but owes
        // the same agreement between the two boxes.
        let lanes = comptime!(fill_extent(&space, sw, w, check));
        let src_rank = comptime!(src.projection.physical_rank());
        let padding = comptime!((sw != w).then(|| {
            // `source_lane` swaps the innermost entry of a destination coordinate to address the
            // source, which only lands on a source cell when the two boxes have the same rank. A
            // storage-tiled stage splits each axis into a grid and a block digit and does not.
            assert!(
                src_rank == plen,
                "MemData::fill_straight: a padded stage is a rank-{plen} box filled from a \
                 rank-{src_rank} source, so a destination coordinate does not address it"
            );
            Padding {
                width: w,
                lanes,
                rank: src_rank,
            }
        }));
        // A comptime worker count emits the tasks straight-line: a rolled loop's runtime `CUBE_DIM`
        // stride blocks unrolling, and on Metal's in-order pipe each line's store then stalls the
        // next line's read. Only a spilling last task needs its guard; unknown or tiny cubes take
        // the rolled loop. `constant()` bridges the folded total back to host data; a whole smem
        // stage's shape is static, so it always folds.
        let units = comptime!(self.access.stage.units);
        let total_c = total.constant();
        // The other half of the fill's contract: the mappings agree ([`stage_compaction`]), and so
        // do the sizes. A gathered destination is always an smem stage, so its line count folds and
        // has to be exactly the compacted window's.
        let cells = comptime!(compaction.as_ref().map(|c| c.cells(w)));
        comptime!(assert!(
            match cells {
                Some(n) => matches!(total_c, Some(t) if t as usize == n),
                None => true,
            },
            "MemData::fill_straight: a gathered source fills a destination of {total_c:?} lines, \
             but its compacted window is {cells:?}"
        ));
        let straight =
            comptime!(matches!(total_c, Some(t) if units > 0 && (t as usize).div_ceil(units) <= 8));
        let d = self.lines_storage_mut::<I2, WP2>();
        if comptime!(sw == w) {
            let s = if comptime!(steps.is_empty()) {
                MaskedView::new(
                    src.lines_storage::<I2, WP2>()
                        .view(src.base())
                        .view(src.window()),
                    check,
                )
            } else {
                MaskedView::new(
                    src.lines_storage::<I2, WP2>()
                        .view(src.base())
                        .view(src.window())
                        .view(StepUp::new(shape.clone(), comptime!(steps))),
                    check,
                )
            };
            fill_lines::<I2, WP2, WP2>(
                d, &s, projection, &shape, total, total_c, units, straight, padding,
            );
        } else {
            let s = if comptime!(steps.is_empty()) {
                MaskedView::new(
                    src.lines_storage::<I2, Const<1>>()
                        .view(src.base())
                        .view(src.window()),
                    check,
                )
            } else {
                MaskedView::new(
                    src.lines_storage::<I2, Const<1>>()
                        .view(src.base())
                        .view(src.window())
                        .view(StepUp::new(
                            widened_shape(&shape, comptime!(plen), comptime!(w)),
                            comptime!(steps),
                        )),
                    check,
                )
            };
            fill_lines::<I2, WP2, Const<1>>(
                d, &s, projection, &shape, total, total_c, units, straight, padding,
            );
        }
    }

    /// Refill this quantized stage's scales side-channel from `src`: one f32 per block of the
    /// sub-tile, into the compact self-relative grid [`smem_quant_info`] laid out, cooperatively
    /// across the cube. The destination index is the flat block index (the grid is row-major); the
    /// source dots the block coords with `src`'s scale strides, whose `window_start` carries base.
    fn stage_scales(&mut self, src: &MemData<T>) {
        let dst = self.store.quant.as_mut().unwrap();
        let sinfo = src.store.quant.as_ref().unwrap();
        let nb = comptime!(dst.scale_shape.clone());
        let rank = comptime!(nb.len());
        let count = comptime!(nb.iter().product::<usize>());
        let dend = dst.buffer.len();
        let dst_scales = dst.buffer.slice_mut(0, dend);
        let send = sinfo.buffer.len();
        let src_scales = sinfo.buffer.slice(0, send);
        let workers = CUBE_DIM as usize;
        let mut bl = UNIT_POS as usize;
        while bl < count {
            let x = bl.fcast::<u32>();
            let mut src_idx = sinfo.window_start;
            #[unroll]
            for p in 0..rank {
                let after = comptime!(nb[(p + 1)..].iter().product::<usize>());
                let bi = x
                    .fdiv(comptime!(after as u32))
                    .frem(comptime!(nb[p] as u32));
                src_idx = src_idx.fadd(bi.fmul(sinfo.strides.at(p)));
            }
            // The grid holds *effective* scales: a two-level source's global level folds in here,
            // once per block per stage, so everything below the stage serves a one-level scheme
            // and no global scale threads past this point.
            dst_scales[bl] = sinfo.known.effective(src_scales[src_idx.fcast::<usize>()]);
            bl += workers;
        }
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

    /// The cooperative flat scan behind [`fill_from`](MemData::fill_from)'s general path: cyclic
    /// across the cube, each unit writing lines `u`, `u + CUBE_DIM`, …. Reads through
    /// [`flat_transparent`](MemData::flat_transparent) at storage element `I`, so a quantized
    /// source dequantizes into `T` transparently (`I == T` on a plain source).
    fn scan_transparent<I: Numeric, WP: Size, W: Size>(&mut self, src: &MemData<T>) {
        let s = src.flat_transparent::<I, WP, W>();
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

    /// The sub-word twin of [`scan_transparent`](MemData::scan_transparent): the source's served
    /// line is one whole packed word and each word unpacks into `num_quants / W` lines of this
    /// store's width, which is how a packed operand stages on a device whose vectors cannot cover
    /// a word. One line **is** one word, so no other width plays.
    ///
    /// Unchecked only and unreachable any other way (a checked operand cannot vectorize), so the
    /// assert below is a backstop for hand-built args; the ragged tail is the caller's ordinary
    /// `checked(false)` claim. The innermost scale block must cover whole words.
    fn scan_words<W: Size>(&mut self, src: &MemData<T>) {
        #[comptime]
        match &src.store.quant {
            ComptimeOption::Some(info) => {
                let nq = comptime!(info.scheme.num_quants());
                comptime!(assert!(
                    src.store.vector_size == nq,
                    "MemData::scan_words: the source serves whole words (vector_size == num_quants)"
                ));
                let w = comptime!(self.store.vector_size);
                comptime!(assert!(
                    w < nq && nq.is_multiple_of(w),
                    "MemData::scan_words: the stage width must divide the packing factor"
                ));
                comptime!(assert!(
                    !src.access.overhang.masks(),
                    "MemData::scan_words: a sub-word fill reads unchecked"
                ));
                comptime!(assert!(
                    info.block.last().unwrap().is_multiple_of(nq),
                    "MemData::scan_words: the innermost scale block must cover whole words"
                ));
                let lpw = comptime!(nq / w);
                let size!(NW) = 1usize;
                let words = src
                    .lines_storage::<u32, NW>()
                    .view(src.base())
                    .view(src.window())
                    .view(FlatLayout::new(src.window.extent.clone()));
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(src.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(FlatLayout::new(src.window.extent.clone()));
                let mut d = self.flat_mut::<W>();
                let total = d.shape();
                let workers = CUBE_DIM as usize;
                let mut i = UNIT_POS as usize;
                while i < total {
                    let word = words.read(i / lpw).extract(0usize);
                    let scale = scales.read(i / lpw);
                    let first = ((i % lpw) * w) as u32;
                    let vals = unpack_fields::<T, W>(
                        word,
                        first,
                        info.table.clone(),
                        comptime!(info.scheme),
                    );
                    d.write(i, vals * Vector::new(T::cast_from(scale)));
                    i += workers;
                }
            }
            ComptimeOption::None => {
                panic!("MemData::scan_words: a plain source has no words to unpack")
            }
        }
    }

    /// Where this operand lives at each level below, and how a materialized level lays its buffer
    /// out; carried from the operand's [`TileSpec`].
    pub(crate) fn stage_plan(&self) -> comptime_type!(StagePlan) {
        comptime!(self.access.stage.clone())
    }

    /// How far this store's quantized form travels ([`DequantAt`]). A plain store answers
    /// [`DequantAt::Load`]: served and stored are the same element, so nothing is left to decode.
    // The `let`-then-return is load-bearing, see [`quant_pack`](MemData::quant_pack).
    #[allow(clippy::let_and_return)]
    pub(crate) fn dequant_at(&self) -> comptime_type!(DequantAt) {
        let dequant_at = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => comptime!(info.dequant_at),
            ComptimeOption::None => DequantAt::Load,
        };
        dequant_at
    }

    /// How this store's values sit in memory, as stated at construction.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(self.store.packing)
    }

    /// This buffer's byte length, widened by the physical width: the transaction count a TMA fill
    /// into it lands. A quantized buffer widens by the *storage* element and packed width instead,
    /// same line count. Unreachable for quant today, but computed rather than refused.
    pub(crate) fn size_bytes(&self) -> u32 {
        let lines = self.store.buffer().len() as u32;
        #[comptime]
        match &self.store.quant {
            ComptimeOption::None => {
                lines * T::size().comptime() as u32 * self.store.vector_size.comptime() as u32
            }
            ComptimeOption::Some(info) => {
                let wp = comptime!(self.store.vector_size / info.scheme.num_quants());
                match comptime!(info.scheme.store) {
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => {
                            lines * i8::size().comptime() as u32 * wp as u32
                        }
                        other => panic!(
                            "MemData::size_bytes: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    QuantStore::PackedU32(_) => lines * u32::size().comptime() as u32 * wp as u32,
                    other => panic!(
                        "MemData::size_bytes: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                }
            }
        }
    }

    /// The base layout: the `[grid…, tile…]` split (`levels > 0`) or a plain
    /// strided dot (`levels = 0`).
    fn base(&self) -> GmemLayout {
        self.layout.clone()
    }

    fn window(&self) -> Window {
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
    /// [`read_view`](MemData::read_view), which an erased source serves and this cannot.
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.store.buffer().as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](MemData::lines). Buffers only: an erased destination has no
    /// address and so no lines to hand out. [`write_view`](MemData::write_view) is the write path
    /// both backings share.
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.store
            .buffer_mut()
            .as_vectorized_mut()
            .with_vector_size_mut::<W>()
    }

    /// The backing as a [`ViewMut`] addressed by `layout`: the write path, and the only one a
    /// [`WriteCall`](Backing::WriteCall) serves. The layout is the same for every backing, which is
    /// the point: only the end of the address differs, a store or a call, so every mutable view
    /// above composes onto this without knowing what it writes to.
    fn write_view<W: Size>(&mut self, layout: GmemLayout) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match &mut self.store.backing {
            Backing::Buffer(buffer) => buffer
                .as_vectorized_mut()
                .with_vector_size_mut::<W>()
                .view_mut(layout),
            Backing::WriteCall(destination) => {
                ViewMut::new::<&mut ErasedTensor<T, WriteOnly>, Coords1d>(destination, layout)
            }
            Backing::ReadCall(_) => panic!(
                "MemData::write_view: this tile's backing is read through a call, which is \
                 read-only"
            ),
        }
    }

    /// The backing as a [`View`] addressed by `layout`: the read path, and the only one a
    /// [`ReadCall`](Backing::ReadCall) serves. The mirror of
    /// [`write_view`](MemData::write_view), so a producer with no slice to hand out is still read
    /// where a buffer is. The slice-shaped half (dense runs, quantized re-typing, tma maps) is
    /// deliberately left out: none is a view over `Coords1d`, so each keeps saying so through
    /// [`Store::buffer`].
    fn read_view<W: Size>(&self, layout: GmemLayout) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.store.backing {
            Backing::Buffer(buffer) => buffer.as_vectorized().with_vector_size::<W>().view(layout),
            Backing::ReadCall(producer) => {
                View::new::<&ErasedTensor<T, ReadOnly>, Coords1d>(producer, layout)
            }
            Backing::WriteCall(_) => panic!(
                "MemData::read_view: this tile's backing is written through a call, and is never \
                 read"
            ),
        }
    }

    /// [`lines`](MemData::lines) with the buffer re-typed to the quantized storage
    /// element `I` it truly holds (see [`QuantInfo`]).
    fn lines_storage<I: Numeric, W: Size>(&self) -> &[Vector<I, W>] {
        let storage = unsafe { self.store.buffer().downcast_unchecked::<I>() };
        storage.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines_storage`](MemData::lines_storage): where a quant stage's
    /// [`fill_straight`](MemData::fill_straight) writes the packed storage words. `I == T` on a
    /// plain copy, a same-type reinterpret.
    fn lines_storage_mut<I: Numeric, W: Size>(&mut self) -> &mut [Vector<I, W>] {
        let storage = unsafe { self.store.buffer_mut().downcast_mut_unchecked::<I>() };
        storage.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// The window as one dense run of lines: index `i` addresses line `origin + i`, one add and no
    /// layout walk. Legal only where the window is physically contiguous in row-major order: an
    /// untiled, unmasked, unquantized store. The comptime-checkable parts assert; contiguity is
    /// the caller's guarantee.
    pub(crate) fn dense_lines<W: Size>(&self) -> &[Vector<T, W>] {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::dense_lines: a dense window cannot mask an overhang"
        ));
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::dense_lines: a storage-tiled window is not dense"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::dense_lines: a gathered window is not dense (sibling windows overlap)"
        ));
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("MemData::dense_lines: a packed store is served through Tile::copy_from")
        }
        let all = self.lines::<W>();
        let start = self.window_start.fcast::<usize>();
        all.slice(start, all.len())
    }

    /// The mutable twin of [`dense_lines`](MemData::dense_lines).
    pub(crate) fn dense_lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::dense_lines_mut: a dense window cannot mask an overhang"
        ));
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::dense_lines_mut: a storage-tiled window is not dense"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::dense_lines_mut: a gathered window is not dense (sibling windows overlap)"
        ));
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("MemData::dense_lines_mut: a packed store cannot be written dense")
        }
        let start = self.window_start.fcast::<usize>();
        let all = self.lines_mut::<W>();
        let end = all.len();
        all.slice_mut(start, end)
    }

    /// The buffer from this window's origin on: the base a cmma load/store addresses,
    /// rows stepping by the scalar [`row_stride`](MemData::row_stride) (cmma takes a line
    /// slice with a scalar stride). Requires an unmasked store whose window doesn't split
    /// rows across storage tiles.
    pub(crate) fn window_slice(&self) -> &[T] {
        let offset = self.window_offset();
        self.store.buffer().slice(offset, self.store.buffer().len())
    }

    /// The mutable twin of [`window_slice`](MemData::window_slice).
    pub(crate) fn window_slice_mut(&mut self) -> &mut [T] {
        let offset = self.window_offset();
        let end = self.store.buffer().len();
        self.store.buffer_mut().slice_mut(offset, end)
    }

    /// Line offset of the window origin: the accumulated `window_start`. On a tiled
    /// store the window must lie within one storage tile.
    fn window_offset(&self) -> usize {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::window_offset: cmma cannot mask an overhang"
        ));
        // A raw window serves the buffer at the element it was erased to, so a quantized store
        // would hand its stored bytes over as served values. Every other door refuses the same way.
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "MemData::window_slice: a packed store has no raw element window; a fragment \
                 load reads it through Tile::matrix_transparent"
            )
        }
        self.window_start.fcast::<usize>()
    }

    /// Scalar stride between matrix rows: the line-unit physical stride of the leaf
    /// tile's row axis, widened back to scalars; a constant on a static store.
    pub(crate) fn row_stride(&self) -> u32 {
        let rank = comptime!(self.layout.projection.physical_rank());
        self.layout
            .physical_strides
            .at(comptime!(rank - 2))
            .fmul(comptime!(self.store.vector_size as u32).runtime())
    }

    /// Re-view this buffer through `layout` as a [`MaskedView`], carrying its own `check` flag
    /// so the leaf masks without being asked. `layout` is a [`TileMatrix`] for the 2-D matmul
    /// leaves and an [`AxisProjection`] for a gathered N-D read.
    pub(crate) fn masked<W: Size, C: Coordinates, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "Tile::matrix: a packed tile only serves values its read unpacks \
                 (Tile::matrix_transparent)"
            )
        }
        MaskedView::new(
            self.read_view::<W>(self.base())
                .view(self.window().with_guard(guard))
                .view(layout),
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
            "MemData: a Boundary::Clamp operand is read-only, a clamped write aliases the edge cell"
        ));
        comptime!(self.access.overhang.masks())
    }

    /// The mutable twin of [`masked`](MemData::masked).
    pub(crate) fn masked_mut<W: Size, C: Coordinates, L: TileLayout<C>>(
        &mut self,
        layout: L,
    ) -> MaskedViewMut<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("Tile::matrix_mut: writing a packed tile requires repacking")
        }
        let base = self.base();
        let window = self.window();
        let check = self.write_check();
        MaskedViewMut::new(
            self.write_view::<W>(base).view_mut(window).view_mut(layout),
            check,
        )
    }

    /// Re-view this buffer as a flat 1-D [`FlatView`] over its [`Window`] extent,
    /// carrying the `check` flag so a flat scan masks the overhang without being asked.
    pub(crate) fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        FlatView::new(
            self.read_view::<W>(self.base())
                .view(self.window())
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// Quantization-transparent [`flat`](MemData::flat): a plain store is read as it stands, a
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
                    .lines_storage::<I, WP>()
                    .view(self.base())
                    .view(self.window())
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

    /// Quantization-transparent [`masked`](MemData::masked): the windowed twin of
    /// [`flat_transparent`](MemData::flat_transparent). A quantized store re-types to the storage
    /// element `I`, pairs it with the scales over the same `layout` and dequantizes each read, so
    /// a leaf reads a quantized operand straight from gmem. `#[comptime]`, so plain pays nothing.
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
    ) -> MaskedView<'_, Vector<T, W>, C> {
        #[comptime]
        match &self.store.quant {
            // A quantized view *is* a view: cubecl's decodes on read and answers as `Vector<T, W>`,
            // so both arms hand back the same masked view and no caller learns the difference.
            ComptimeOption::Some(info) => {
                // The storage view groups at the *physical* width: a packed buffer holds
                // `W / num_quants` elements per served line.
                let values = self
                    .lines_storage::<I, WP>()
                    .view(self.base())
                    .view(self.window().with_guard(guard))
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
                MaskedView::new(
                    dequant.view(),
                    comptime!(guard.checks() && self.access.overhang.masks()),
                )
            }
            ComptimeOption::None => self.unscaled::<WP, W, C, L>(layout, guard),
        }
    }

    /// The scale-free half of [`transparent`](MemData::transparent): a plain store read as it
    /// stands, a packed one unpacked at the read ([`PackedView`]). `WP` is the physical line the
    /// buffer holds, `W` the served one. Needs the field alone where the quantized arm needs a
    /// scheme, a grid and a window start: a packed operand's values are values.
    fn unscaled<WP: Size, W: Size, C: Coordinates + 'static, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, C> {
        let packing = self.packing();
        match comptime!(packing) {
            Packing::Plain => self.masked::<W, C, L>(layout, guard),
            Packing::Native => panic!(
                "MemData::transparent: a native store with nothing to fold in serves its own \
                 element; bind it as that element and let the contraction cast it"
            ),
            Packing::Packed { field } => {
                let words = self
                    .lines_storage::<u32, WP>()
                    .view(self.base())
                    .view(self.window())
                    .view(layout);
                let values = PackedView::<WP, T, W, C>::new(words, comptime!(field));
                MaskedView::new(
                    values.view(),
                    comptime!(guard.checks() && self.access.overhang.masks()),
                )
            }
        }
    }

    /// [`transparent`](MemData::transparent) over one batch matrix: what the 2-D matmul leaves
    /// read. `L` is [`TileMatrix`] for a direct operand and
    /// [`ProjectedMatrix`](super::ProjectedMatrix) for a gathered one; both answer the
    /// same [`Coords2d`] surface.
    pub(crate) fn matrix_transparent<I: Numeric, WP: Size, W: Size, L: TileLayout<Coords2d>>(
        &self,
        layout: L,
    ) -> MatrixView<'_, Vector<T, W>> {
        self.transparent::<I, WP, W, Coords2d, L>(layout, comptime!(Guard::Checked))
    }

    /// [`transparent`](MemData::transparent) over the tile's whole logical box, applying the
    /// operand's [`Projection`]: what a gather-reduce leaf reads, one coordinate per axis.
    pub(crate) fn nd_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
        layout: AxisProjection,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        self.transparent::<I, WP, W, CoordsDyn, AxisProjection>(layout, guard)
    }

    /// [`nd_transparent`](MemData::nd_transparent) over the *physical* box instead of the logical
    /// one, for a caller that folds the map itself. The map is the only layer dropped: the
    /// [`Window`] owning the boundary sits below either way, and the identity step keeps the box's
    /// own bound, which is the part a caller cannot fold away.
    ///
    /// The logical box test goes with the map, though, so this masks against the physical box
    /// alone. A caller owes it the coordinates the map would have produced from inside the logical
    /// one: a position folded from a logical coordinate out of range is no longer caught here, and
    /// reads whatever the window says lives at it.
    pub(crate) fn nd_physical<I: Numeric, WP: Size, W: Size>(
        &self,
    ) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        let rank = comptime!(self.projection.physical_rank());
        let identity = StepUp::new(self.window.extent.clone(), comptime!(vec![1; rank]));
        // A folded caller hands in coordinates it derived itself, so it has proved nothing about
        // them: the window's boundary and the overhang mask both stay on.
        self.transparent::<I, WP, W, CoordsDyn, StepUp>(identity, comptime!(Guard::Checked))
    }

    /// The mutable twin of [`flat`](MemData::flat).
    pub(crate) fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("Tile::flat_mut: writing a packed tile requires repacking")
        }
        let base = self.base();
        let window = self.window();
        let extent = self.window.extent.clone();
        let check = self.write_check();
        FlatViewMut::new(
            self.write_view::<W>(base)
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
        #[comptime] axes: MatrixAxes,
        #[comptime] space: Space,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        // A write aliases only where two logical positions share a cell, which is what an
        // overlapping map is; a partition is a bijection, so its windows tile and each cell is
        // written once. A gathered operand is read through `Tile::nd` and never written here.
        comptime!(assert!(
            self.projection.composition() != Composition::Overlapping,
            "MemData::matrix_mut: an overlapping operand aliases under a write"
        ));
        // Leading (batch) extents are width-invariant; the window extent is the view's shape.
        let bound = self.extent();
        let layout = projected_batch_matrix(
            &bound,
            comptime!(space.clone()),
            comptime!(self.projection.clone()),
            self.map.clone(),
            comptime!(self.store.vector_size),
            axes,
            i,
        );
        self.masked_mut::<W, Coords2d, ProjectedMatrix>(layout)
    }

    /// The [`AccumulateView`] over batch matrix `i`: [`matrix_mut`](MemData::matrix_mut) plus the
    /// [`LaneShare`] these cells carry, the [`Monoid`] they fold under and what the accumulation
    /// starts from, so a leaf accumulates through it without being told any of the three.
    pub(crate) fn matrix_accumulate<W: Size>(
        &mut self,
        i: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] space: Space,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W> {
        let lanes = comptime!(self.lanes);
        let split_share = comptime!(self.split_share);
        let write = comptime!(self.access.write);
        let init_from = comptime!(self.init_from);
        AccumulateView::new(
            self.matrix_mut::<W>(i, axes, space),
            lanes,
            split_share,
            write,
            monoid,
            init_from,
        )
    }

    /// The [`AccumulateView`] over flat elements: [`flat_mut`](MemData::flat_mut) plus the
    /// [`LaneShare`] these cells carry and the [`Monoid`] they fold under.
    pub(crate) fn flat_accumulate<W: Size>(
        &mut self,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W, Coords1d> {
        // A flat logical scan only agrees with this physical window under the direct,
        // non-storage-tiled mapping. Otherwise the reduction's logical accumulator index would
        // seed and commit a different physical cell than the one it reduces for.
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::flat_accumulate: a storage-tiled window has no flat logical accumulator view"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::flat_accumulate: a gathered window has no flat logical accumulator view"
        ));
        let lanes = comptime!(self.lanes);
        let split_share = comptime!(self.split_share);
        let write = comptime!(self.access.write);
        let init_from = comptime!(self.init_from);
        AccumulateView::new(
            self.flat_mut::<W>(),
            lanes,
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
    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> MemData<T> {
        let mut origin = Coords::<i32>::new();
        let mut extent = Coords::<u32>::new();
        // Per-physical-axis window_start advances, summed below (chained, so constants fold).
        let mut advances = Coords::<u32>::new();

        let proj = comptime!(self.projection.clone());
        let rank = comptime!(proj.physical_rank());
        let last = comptime!(rank - 1);
        let w = comptime!(self.store.vector_size);

        let map = if comptime!(proj.is_direct()) {
            // One logical axis per physical axis at coefficient 1. Kept as its own loop because
            // this is the only mapping a *tiled* buffer can carry, where `step` folds the
            // grid/tile digit split that a scaled advance cannot be pushed through.
            #[unroll]
            for p in 0..rank {
                let axis = space.axis_at(p);
                // The innermost (vectorized) axis's edge is a line count, so `/ width`.
                let edge = comptime!(if p == last {
                    let e = space.partitioner().edge(axis);
                    // A padded stage's innermost extent need not fill whole lines, but then its
                    // partial tail line has no sibling to start after it: the axis has to be cut
                    // whole, or the next region would begin mid-line. `extent_raw` because a
                    // `Dynamic` axis has no extent to be cut whole, and owes the divisibility.
                    assert!(
                        e.is_multiple_of(w)
                            || matches!(space.extent_raw(axis), Extent::Static(x) if x == e),
                        "MemData::at: the innermost edge {e} is neither a whole number of \
                         {w}-wide lines nor the axis's whole extent ({:?}), so a region would \
                         start mid-line",
                        space.extent_raw(axis)
                    );
                    e.div_ceil(w)
                } else {
                    space.partitioner().edge(axis)
                });
                let index = region.coord(axis);

                origin.push(
                    self.window
                        .origin
                        .at(p)
                        .fadd(index.fmul(edge).fcast::<u32>().fcast::<i32>()),
                );
                extent.push(comptime!(edge as u32).runtime());
                advances.push(index.fcast::<u32>().fmul(step_offset(
                    comptime!(self.layout.projection.clone()),
                    comptime!(Axis(p as u8)),
                    edge,
                    &self.layout.physical_shape,
                    &self.layout.physical_strides,
                )));
            }
            // Every axis at coefficient 1, which no Dynamic term and no divisor can spell, so
            // there is nothing to carry and no phase to leave over.
            RuntimeMap::integral(rank)
        } else {
            let mut residues = Coords::<u32>::new();
            #[unroll]
            for pa in 0..rank {
                let (step, residue, span) = gathered_descent(
                    comptime!(proj.clone()),
                    comptime!(space.clone()),
                    region,
                    &self.map,
                    w,
                    pa,
                );

                // `step` only moves forward, so add directly to the signed origin.
                origin.push(self.window.origin.at(pa).fadd(step.fcast::<i32>()));
                residues.push(residue);
                extent.push(span);
                // `Projection::validate` pins a gathered operand to untiled storage (bare gmem, or
                // the row-major compacted stage of one), so one physical axis step is one stride
                // and the advance passes straight through.
                advances.push(step.fmul(self.layout.physical_strides.at(pa)));
            }
            // The coefficients are a fact about the buffer, invariant down the descent; only the
            // phase each axis's division left over is this level's.
            RuntimeMap {
                coefficients: self.map.coefficients.clone(),
                residues,
            }
        };
        let start = self
            .window_start
            .fadd(advances.fsum(comptime!((0..rank).collect::<Vec<_>>())));

        // Re-window the scales alongside the values.
        let mut origin_u32 = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            origin_u32.push(origin.at(p).fcast::<u32>());
        }
        let quant = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => {
                comptime!(assert!(
                    !self.window.signed,
                    "MemData::at: a quantized operand cannot carry a negative window origin, its \
                     scale grid is addressed unsigned"
                ));
                // A quantized operand is direct (asserted at construction), so the child window's
                // extent per axis is this level's cut edge.
                ComptimeOption::new_Some(info.window(
                    &origin_u32,
                    rank,
                    comptime!(self.store.vector_size),
                    comptime!(
                        (0..rank)
                            .map(|p| space.partitioner().edge(space.axis_at(p)))
                            .collect::<Vec<_>>()
                    ),
                ))
            }
            ComptimeOption::None => ComptimeOption::new_None(),
        };

        MemData::<T> {
            store: Store::<T> {
                backing: self.store.backing.clone(),
                vector_size: comptime!(self.store.vector_size),
                quant,
                packing: comptime!(self.store.packing),
            },
            // The layout addresses the whole buffer and never narrows; only the window moves.
            layout: self.layout.clone(),
            window: Window::new(
                origin,
                extent,
                self.window.bound.clone(),
                comptime!(self.window.signed),
                comptime!(self.window.boundaries.clone()),
            ),
            // How the logical axes address the physical ones is a fact about the buffer, invariant
            // down the descent. The offsets only placed the top window, which `origin` above
            // already carries.
            projection: comptime!(proj),
            // A region step moves this window and the source window by the same physical delta,
            // so the source window rides down as it was filled and only `origin` above moves.
            source_window: self.source_window.clone(),
            map,
            offsets: self.offsets.clone(),
            window_start: start,
            // The window no longer covers the buffer, so the straight-through fill is off. The
            // plan descends with the space: this level's residence is behind us now.
            access: comptime!(Access {
                whole: false,
                overhang: self.access.overhang,
                write: self.access.write,
                stage: self.access.stage.descend(),
            }),
            lanes: comptime!(Lanes {
                share: join_lane_share(self.lanes.share, space.lane_share()),
                work: self.lanes.work,
            }),
            // Settled at construction, so a descent carries it: which instances hold a cell is a
            // fact about the whole space, and windowing down does not change it.
            split_share: comptime!(self.split_share),
            init_from: comptime!(self.init_from),
        }
    }
}

/// One gathered physical axis's descent into `region`: how far its window moves, the phase that
/// move leaves behind, and the receptive field the child then covers.
///
/// The move sums one term per contributing axis, its tile coordinate times its sub-tile edge times
/// its coefficient. All comptime but the coordinate, so this stays the multiply-add `window_start`
/// is documented to be. Each term divides by `vector_size` on its own, which only sums back to the
/// whole move because the innermost physical axis carries a single identity term:
/// [`Projection::validate`] requires it, precisely because that axis is addressed in lines. A
/// second term here would need the division after the sum, not before.
///
/// A [rational](crate::Divisor) axis moves by the whole cells its numerator crossed and hands the
/// phase it did not fill to the child: `⌊(move + phase)/d⌋` splits into this step plus a child
/// floor starting at the new phase, which is what makes the descent compose across levels.
#[cube]
fn gathered_descent(
    #[comptime] projection: Projection,
    #[comptime] space: Space,
    region: &Region,
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
        let edge = comptime!(space.partitioner().edge(term.axis));
        match comptime!(term.scale) {
            Scale::Static(s) => {
                let step = comptime!(if lined {
                    edge * s / vector_size
                } else {
                    edge * s
                });
                terms.push(region.coord(term.axis).fmul(step).fcast::<u32>());
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
                    region
                        .coord(term.axis)
                        .fcast::<u32>()
                        .fmul(comptime!(edge as u32).runtime())
                        .fmul(coefficient),
                );
                spans.push(comptime!((edge - 1) as u32).runtime().fmul(coefficient));
            }
        }
    }
    let advance = terms.fsum(comptime!(picks.clone()));

    if comptime!(!axis_map.is_rational()) {
        // The receptive field of the child edges: `1 + Σ (edge - 1) * scale`, which stays comptime
        // for the mapping that is.
        let span = if comptime!(!axis_map.has_dynamic_scale()) {
            comptime!({
                let s = projection.span(pa, |a| space.partitioner().edge(a));
                (if lined { s / vector_size } else { s }) as u32
            })
            .runtime()
        } else {
            spans.fsum(comptime!(picks.clone())).fadd(1)
        };
        (advance, 0u32, span)
    } else {
        // No `/ vector_size` anywhere below, and none is owed: `Projection::validate` refuses a
        // rational innermost physical axis at any width past `1`, so it is `1` whenever this
        // branch runs and the terms above are already in elements.
        let numerator = advance.fadd(map.residues.at(pa));
        let field = spans.fsum(comptime!(picks.clone()));
        match comptime!(axis_map.divisor()) {
            Divisor::Static(d) => {
                let d = comptime!(d as u32);
                let residue = numerator.frem(d);
                (
                    numerator.fdiv(d),
                    residue,
                    field.fadd(residue).fdiv(d).fadd(1),
                )
            }
            Divisor::Dynamic { .. } => {
                let d = map
                    .coefficients
                    .at(comptime!(projection.dynamic_divisor_index(pa).unwrap()));
                let residue = numerator.frem(d);
                (
                    numerator.fdiv(d),
                    residue,
                    field.fadd(residue).fdiv(d).fadd(1),
                )
            }
        }
    }
}

/// The innermost extent of `space` in cells, with the two widths a fill pairs checked against it.
///
/// The fill reads whole `sw`-wide source lines, so the innermost extent has to be a whole number
/// of them, and only the *destination* may hold a partial `w`-wide one. That partial line is what
/// a padded stage is, and its spare lanes hold zero. Without this the two boxes silently disagree,
/// the stage rounding its line count up ([`storage_extents`], `Compaction::line_extents`) where
/// the source truncated its own.
///
/// `None` for a `Dynamic` extent: nothing can be said at comptime, so a padded stage over one
/// leans on `check` to zero its spare lanes instead.
fn fill_extent(space: &Space, sw: usize, w: usize, check: bool) -> Option<usize> {
    match space.extent_raw(space.axis_at(space.rank() - 1)) {
        Extent::Static(e) => {
            assert!(
                e.is_multiple_of(sw),
                "MemData: the innermost extent {e} is not a whole number of the source's \
                 {sw}-wide lines, so the stage holds cells the source cannot hand it"
            );
            Some(e)
        }
        Extent::Dynamic => {
            assert!(
                sw == w || check,
                "MemData: a padded stage over a Dynamic innermost extent cannot know at comptime \
                 which lanes are padding, so its source must be bounds-checked for them to read \
                 as zero"
            );
            None
        }
    }
}

/// Schedule cooperative cyclic writing of destination stage lines across cube units.
///
/// Dispatches line reads via [`read_stage_line`], taking an unrolled loop when the task count
/// is small and static (`straight == true`) or a dynamic `CUBE_DIM`-strided while loop otherwise.
#[cube]
fn fill_lines<I2: Numeric, WP2: Size, SW: Size>(
    d: &mut [Vector<I2, WP2>],
    s: &MaskedView<'_, Vector<I2, SW>, CoordsDyn>,
    #[comptime] projection: Projection,
    shape: &Coords<u32>,
    total: usize,
    #[comptime] total_c: Option<u64>,
    #[comptime] units: usize,
    #[comptime] straight: bool,
    #[comptime] padding: Option<Padding>,
) {
    if comptime!(straight) {
        let tasks = comptime!((total_c.unwrap() as usize).div_ceil(units));
        #[unroll]
        for t in 0..tasks {
            let i = UNIT_POS as usize + comptime!(t * units);
            if comptime!((t + 1) * units > total_c.unwrap() as usize) {
                if i < total {
                    d[i] = read_stage_line::<I2, WP2, SW>(
                        s,
                        &physical_pos(comptime!(projection.clone()), i, shape),
                        comptime!(padding),
                    );
                }
            } else {
                d[i] = read_stage_line::<I2, WP2, SW>(
                    s,
                    &physical_pos(comptime!(projection.clone()), i, shape),
                    comptime!(padding),
                );
            }
        }
    } else {
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            d[i] = read_stage_line::<I2, WP2, SW>(
                s,
                &physical_pos(comptime!(projection.clone()), i, shape),
                comptime!(padding),
            );
            i += workers;
        }
    }
}

/// Read one destination line from the masked source view at `pos`: whole for a 1:1 copy, or
/// assembled lane by lane from scalar source cells for a padded stage ([`widen_line`]).
#[cube]
fn read_stage_line<I2: Numeric, WP2: Size, SW: Size>(
    s: &MaskedView<'_, Vector<I2, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Option<Padding>,
) -> Vector<I2, WP2> {
    if comptime!(padding.is_some()) {
        widen_line::<I2, WP2, SW>(s, pos, comptime!(padding.unwrap()))
    } else {
        // The unpadded caller builds its view at the destination's own width, so `SW` *is* `WP2`
        // here and the cast is an identity the trace folds away; the two only differ as types.
        Vector::<I2, WP2>::cast_from(s.read(pos.clone()))
    }
}

/// The logical coordinate of physical line `i` in a `[grid…, tile…]` store: decode `i` into one
/// digit per physical axis ([`line_digit`]), then [`fold_physical`] folds a storage-tiled axis's
/// several digits back into one, off `projection`'s own div/modulo (`GmemLayout`'s synthetic
/// per-position map, invertible by construction).
#[cube]
fn physical_pos(#[comptime] projection: Projection, i: usize, shape: &Coords<u32>) -> CoordsDyn {
    let x = i.fcast::<u32>();
    let mut digits = Coords::<u32>::new();
    #[unroll]
    for j in 0..shape.len() {
        digits.push(line_digit(x, shape, j));
    }
    fold_physical(comptime!(projection), &digits, shape)
}

/// Assemble one padded destination line from adjacent scalar source cells.
///
/// When `Padding::lanes` is `None` (a `Dynamic` innermost extent), the source window must be
/// bounds-checked so that reads past the extent return zero. When it is `Some(n)`, reads past `n`
/// are masked off explicitly so the padding lanes keep the zero they start at.
#[cube]
fn widen_line<T: Numeric, W: Size, SW: Size>(
    s: &MaskedView<'_, Vector<T, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Padding,
) -> Vector<T, W> {
    let width = comptime!(padding.width);
    let rank = comptime!(padding.rank);
    comptime!(assert!(
        SW::try_value_const() == Some(1),
        "widen_line: a padded stage is filled from a scalar source, got a {:?}-wide one",
        SW::try_value_const()
    ));
    comptime!(assert!(
        W::try_value_const().is_none_or(|n| n == width),
        "widen_line: assembles {width} lanes into a {:?}-wide destination line",
        W::try_value_const()
    ));
    let last = comptime!(rank - 1);
    let line = pos[last];
    let mut out = Vector::<T, W>::cast_from(T::from_int(0));
    let guarded = comptime!(match padding.lanes {
        Some(n) => !n.is_multiple_of(width),
        None => false,
    });
    #[unroll]
    for l in 0..width {
        let cell = line.fmul(comptime!(width as u32)).fadd(comptime!(l as u32));
        let valid = if comptime!(guarded) {
            cell < comptime!(padding.lanes.unwrap() as u32)
        } else {
            true.runtime()
        };
        if valid {
            out.insert(
                l,
                s.read(source_lane(pos, comptime!(rank), cell))
                    .extract(0usize),
            );
        }
    }
    out
}

/// Replace the destination line coordinate with its scalar source-cell coordinate.
#[cube]
fn source_lane(pos: &CoordsDyn, #[comptime] rank: usize, cell: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(cell);
        } else {
            out.push(pos[p]);
        }
    }
    out
}

/// Express a padded destination's innermost physical extent in scalar source elements.
#[cube]
fn widened_shape(
    shape: &Coords<u32>,
    #[comptime] rank: usize,
    #[comptime] width: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(shape.at(p).fmul(comptime!(width as u32)));
        } else {
            out.push(shape.at(p));
        }
    }
    out
}

/// Digit `j` of flat line `x` under `shape`'s row-major suffix strides.
#[cube]
fn line_digit(x: u32, shape: &Coords<u32>, #[comptime] j: usize) -> u32 {
    let plen = shape.len();
    x.fdiv(shape.fproduct(comptime!(((j + 1)..plen).collect::<Vec<_>>())))
        .frem(shape.at(j))
}
