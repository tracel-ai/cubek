//! The addressable backing store ([`MemData`], gmem and smem): its layouts, windows, and
//! the memory-side [`Tile`] operations (construction, views, the cooperative copy).

use cubecl::{
    prelude::*,
    quant::scheme::{QuantLevel, QuantScheme, QuantStore, QuantValue},
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, View, ViewMut,
        layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
    },
};

use crate::*;

/// A lifetime-erased buffer, how to address it ([`layout`](GmemLayout)), and which part of it this
/// tile is looking at ([`window`](Window)). The layout is fixed at construction, so a staged smem
/// sub-tile keeps addressing its whole buffer after [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemData<T: Numeric> {
    /// What the bytes are and mean.
    pub(crate) store: Store<T>,
    /// How a logical coordinate becomes a buffer offset. Fixed at construction.
    pub(crate) layout: GmemLayout,
    /// The region of the logical space this tile covers; narrowed by [`at`](Tile::at).
    pub(crate) window: Window,
    /// The window origin's offset through the layout, accumulated across [`at`](Tile::at)s rather
    /// than re-derived: each descent shifts by a *comptime* edge, so [`step`](MemData::step) folds
    /// and this stays a multiply-add. Addressing it from the origin instead would decompose a
    /// runtime coordinate, i.e. integer division per [`window_slice`](MemData::window_slice).
    window_start: u32,
    /// How this store may be touched. All comptime, all decided at construction.
    #[cube(comptime)]
    pub(crate) access: Access,
    /// What each lane holds of these cells, stamped across [`at`](Tile::at)s (the level that
    /// spreads an axis is consumed on the way down). `Partial` means split to an accumulator but
    /// merely replicated to an operand, so only an accumulator reads it.
    #[cube(comptime)]
    pub(crate) lane_share: LaneShare,
}

/// What a [`MemData`]'s bytes are and mean: the erased buffer, the width it groups into lines at,
/// and — when the buffer physically holds quantized data — how a *stored* value becomes a *served*
/// one. Reads through [`Tile::flat`] dequantize into `T`; every other element view refuses a
/// quantized tile.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Store<T: Numeric> {
    /// Backing store, scalar-typed by Rust-side erasure only: the real binding/alloc element is
    /// `Vector<T, vector_size>` (see [`VecTensor`](crate::VecTensor)), so re-grouping to lines
    /// at that width is a no-op.
    pub(crate) buffer: Box<[T]>,
    /// Physical line size (`Vector<T, vector_size>`) of the backing store, `1` when
    /// unvectorized; held comptime so `size!` can read it.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Present when the buffer holds quantized data (see [`QuantInfo`]).
    pub(crate) quant: ComptimeOption<QuantInfo>,
}

/// How a [`MemData`] may be touched: whether the fill can write straight through, how the store
/// handles overhang, and how a cooperative fill spreads. Plain data held comptime, like the
/// [`StagePlan`] it carries.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Access {
    /// Whether the window still covers the whole buffer (constructors yes, [`at`](Tile::at) no):
    /// such a tile can be written in physical order.
    pub whole: bool,
    pub overhang: Overhang,
    /// How this store's stages are laid out and cooperatively filled: the [`StageStorage`] layout
    /// plus the launch's cube size. Carried from the operand's [`Storage`] so a fill re-derives
    /// neither.
    pub stage: StagePlan,
}

/// How a store relates to the window overhanging its valid data (`origin + pos` past
/// [`Window`]'s `bound`) — where gmem and smem genuinely differ.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Overhang {
    /// Structurally impossible: the buffer is allocated to exactly the tile (smem).
    Never,
    /// Possible in principle, excluded at launch: every shape divides its tiling (unchecked gmem).
    Fits,
    /// Possible: reads past `bound` serve zero, writes skip (checked gmem).
    Masked,
}

impl Overhang {
    /// The flag a [`MaskedView`] is built with — the one place the states collapse to a bool.
    pub fn masks(&self) -> bool {
        matches!(self, Overhang::Masked)
    }
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
    /// `I == T` with `quant == None`; [`StridedTileArg::tile_dequant`] is the kernel-side constructor.
    pub fn from_tensor_quant<I: Numeric>(
        tensor: &VecTensor<I>,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] storage: Storage,
        quant: ComptimeOption<QuantInfo>,
    ) -> Tile<T> {
        // `vector_size` counts *served values*; the binding is grouped at the storage width, which
        // a packed store narrows by its packing factor. The two coincide on every plain operand.
        let bound_width = tensor.vector_size();
        let pack = #[comptime]
        match &quant {
            ComptimeOption::Some(info) => comptime!(info.scheme.num_quants()),
            ComptimeOption::None => 1usize,
        };
        comptime!(assert!(
            bound_width * pack == vector_size,
            "MemData::from_tensor: comptime vector_size ({vector_size}) is not the binding's \
             width ({bound_width}) times the packing factor ({pack})"
        ));
        let start_axis = comptime!(storage.start_axis);
        let num_tiled = comptime!(space.rank() - storage.start_axis);
        let levels = comptime!(storage.levels);
        let rank = comptime!(start_axis + (levels + 1) * num_tiled);
        let last = comptime!(rank - 1);
        let w = comptime!(vector_size as u32);
        let mut physical_shape = Coords::<u32>::new();
        let mut physical_strides = Coords::<u32>::new();
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
                store: Store::<T> {
                    buffer,
                    vector_size: comptime!(vector_size),
                    quant,
                },
                layout: GmemLayout {
                    physical_shape,
                    physical_strides,
                    start_axis,
                    num_tiled,
                    levels,
                },
                window: Window::new(origin, extent, bound),
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang: if storage.check_bounds {
                        Overhang::Masked
                    } else {
                        Overhang::Fits
                    },
                    stage: storage.stage,
                }),
                lane_share: comptime!(LaneShare::Whole),
            }),
            space: comptime!(space),
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of `operand`, in
    /// the element the operand *serves*: a quantized operand dequantizes at the fill, so the stage
    /// holds `T`. The twin is [`smem_like_stored`](MemData::smem_like_stored), which keeps the
    /// operand's stored element instead and dequantizes at the read; a leaf that cannot dequantize
    /// on read (cmma) must take this one. Identical to its twin for a plain operand, whose served
    /// and stored elements are the same.
    pub fn smem_like(operand: &Tile<T>) -> Tile<T> {
        MemData::smem(
            comptime!(operand.space.divide()),
            operand.vector_size(),
            operand.stage(),
        )
    }

    /// [`smem_like`](MemData::smem_like) in the element the operand is *stored* in rather than the
    /// one it serves: a quantized operand keeps its stored form (native `i8`, or `u32` words when
    /// the scheme packs several values each) and its scales, so the leaf dequantizes at the read
    /// (see [`smem_quant`](MemData::smem_quant)) instead of the fill inflating the stage to `T`.
    /// Only a leaf that reads through [`matrix_transparent`](MemData::matrix_transparent) can be
    /// served this; `Staging::new` owns that call.
    pub fn smem_like_stored(operand: &Tile<T>) -> Tile<T> {
        let space = comptime!(operand.space.divide());
        let vector_size = operand.vector_size();
        let stage = operand.stage();
        match &operand.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                #[comptime]
                match &g.store.quant {
                    // Served == stored, so this is `smem_like`.
                    ComptimeOption::None => MemData::smem(space, vector_size, stage),
                    ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                        QuantStore::Native => match comptime!(info.scheme.value) {
                            QuantValue::Q8F | QuantValue::Q8S => MemData::smem_quant::<i8>(
                                space,
                                vector_size,
                                stage,
                                comptime!(info.scheme),
                            ),
                            other => panic!(
                                "MemData::smem_like_stored: native quant storage element {:?} is not wired (i8 only)",
                                other
                            ),
                        },
                        QuantStore::PackedU32(_) => MemData::smem_quant::<u32>(
                            space,
                            vector_size,
                            stage,
                            comptime!(info.scheme),
                        ),
                        other => panic!(
                            "MemData::smem_like_stored: quant storage {:?} is not wired (native or packed-u32)",
                            other
                        ),
                    },
                }
            }
            // A tma source has no stored form to keep: it carries no scheme (`quantized` is a
            // [`StridedTileArg`] builder, and a tma tile is scalar), so served == stored. Giving it
            // one must not reuse this arm — see `Staging::new`, which refuses that combination.
            TileKind::TmaGmem(_) => MemData::smem(space, vector_size, stage),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("MemData::smem_like_stored: a fragment is not a stage source")
            }
        }
    }

    /// Allocate a shared-memory tile over `space`, at physical `vector_size` (the slice is
    /// allocated natively wide, then scalar-erased). A `Tiled` stage is storage-tiled at the
    /// final tile (one contiguous block per fragment, what a cmma transaction wants) so the cmma
    /// transaction reads it unstrided; `Strided` is plain row-major. A final-space stage has no
    /// grid to tile, so it is always plain. `units` is the launch's cube size, `0` when unknown.
    pub fn smem(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
    ) -> Tile<T> {
        let size!(W) = vector_size;
        let smem = Shared::<[Vector<T, W>]>::new_slice(comptime!(space.tile_size() / vector_size));
        MemData::smem_over(space, vector_size, stage, &smem, ComptimeOption::new_None())
    }

    /// [`smem`](MemData::smem) staging the element an operand is *stored* in rather than the one it
    /// serves: `I` is that element (`i8`, or `u32` when the scheme packs several values per word)
    /// and the line narrows to `vector_size / pack`, so the stage is that much smaller and a leaf
    /// dequantizes at the read instead of the fill inflating it to `T`. Carries a compact `Shared`
    /// scales buffer beside the values (one f32 per block of the sub-tile, refilled per region by
    /// [`fill_from`](MemData::fill_from)).
    pub fn smem_quant<I: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
        #[comptime] scheme: QuantScheme,
    ) -> Tile<T> {
        // One stored line is one served line, just narrower, so only the element and width change:
        // the layout and window below are the same grid either way.
        let size!(WP) = comptime!(vector_size / scheme.num_quants());
        let smem = Shared::<[Vector<I, WP>]>::new_slice(comptime!(space.tile_size() / vector_size));
        let quant = smem_quant_info(comptime!(space.clone()), comptime!(scheme));
        MemData::smem_over(space, vector_size, stage, &smem, quant)
    }

    /// The body both smem constructors share: everything but the allocation's element, which is why
    /// `smem` takes the allocated slice rather than making it. Scalar-erases it to the served `T`
    /// (as [`from_tensor_quant`](MemData::from_tensor_quant) does; the views recover the stored
    /// element through [`lines_storage`](MemData::lines_storage)) and wraps it in the whole-buffer
    /// window.
    fn smem_over<S: CubePrimitive>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
        smem: &Shared<[S]>,
        quant: ComptimeOption<QuantInfo>,
    ) -> Tile<T> {
        let levels = comptime!((!space.is_final() && stage.layout == StageStorage::Tiled) as usize);
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
                store: Store::<T> {
                    buffer,
                    vector_size,
                    quant,
                },
                layout: GmemLayout {
                    physical_shape,
                    physical_strides,
                    start_axis: comptime!(0usize),
                    num_tiled: comptime!(space.rank()),
                    levels,
                },
                window: Window::new(origin, extent, bound),
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang: Overhang::Never,
                    stage,
                }),
                lane_share: comptime!(LaneShare::Whole),
            }),
            space: comptime!(space),
        }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A read [`View`] over `Vector<T, W>` lines: the scalar buffer re-grouped into its physical
    /// width, then re-viewed through the base layout and [`Window`]. `W` is the line width
    /// (`self.store.vector_size`); pass `Const<1>` when only the (width-invariant) leading shape is needed.
    pub fn view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                if comptime!(g.store.quant.is_some()) {
                    panic!(
                        "Tile::view: a quantized tile only serves dequantized reads (Tile::flat)"
                    )
                }
                g.lines::<W>().view(g.base()).view(g.window())
            }
            TileKind::TmaGmem(_) => panic!("Tile::view: a tma source has no element view"),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::view: a plane tile has no memory view")
            }
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
                g.lines_mut::<W>().view_mut(base).view_mut(window)
            }
            TileKind::TmaGmem(_) => panic!("Tile::view_mut: a tma source has no element view"),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::view_mut: a plane tile has no memory view")
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
        let size!(W) = comptime!(self.store.vector_size);
        if comptime!(self.store.quant.is_some()) {
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
                        QuantValue::Q8F | QuantValue::Q8S => self.fill_straight::<i8, W>(src),
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
                        self.fill_straight::<u32, WP>(src);
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
            self.access.whole && !self.access.overhang.masks() && src.store.quant.is_none()
        ) {
            // Plain → plain, whole destination: fill in destination-physical order (the write is
            // linear and only the source decodes, once per line by constants on a static store).
            self.fill_straight::<T, W>(src);
        } else {
            // The read decodes at the source's true storage element: `T` for a plain tile, else the
            // quantized store's element recovered from its scheme (the tile serves `T`, so `I` was
            // erased at construction and lives only on the scheme). This is what lets a plain
            // `copy_from`/`fill` dequantize on its own into a plain destination — the kernel never
            // threads `I`.
            #[comptime]
            match &src.store.quant {
                ComptimeOption::None => self.scan_transparent::<T, W, W>(src),
                ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => self.scan_transparent::<i8, W, W>(src),
                        other => panic!(
                            "MemData::fill_from: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    QuantStore::PackedU32(_) => {
                        let size!(WP) = comptime!(src.store.vector_size / info.scheme.num_quants());
                        self.scan_transparent::<u32, WP, W>(src)
                    }
                    other => panic!(
                        "MemData::fill_from: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                },
            }
        }
    }

    /// The straight-line half of [`fill_from`](MemData::fill_from): a whole destination filled in
    /// destination-physical order, whole `Vector<I2, WP2>` lines, only the source decoding (once per
    /// line, by constants on a static store — half the address math of a logical-order scan). `I2` /
    /// `WP2` are the *storage* element and physical width: the served `(T, self.store.vector_size)` for a
    /// plain copy, the packed storage `(u32, served/pack)` (or native `i8`) for a quant stage.
    fn fill_straight<I2: Numeric, WP2: Size>(&mut self, src: &MemData<T>) {
        let s = MaskedView::new(
            src.lines_storage::<I2, WP2>()
                .view(src.base())
                .view(src.window()),
            comptime!(src.access.overhang.masks()),
        );
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        let total = shape
            .fproduct(comptime!((0..plen).collect::<Vec<_>>()))
            .fcast::<usize>();
        let start_axis = comptime!(self.layout.start_axis);
        let num_tiled = comptime!(self.layout.num_tiled);
        let levels = comptime!(self.layout.levels);
        // A comptime worker count emits the tasks straight-line: a rolled loop's runtime `CUBE_DIM`
        // stride blocks unrolling, and on Metal's in-order pipe each line's store then stalls the
        // next line's read. Only a spilling last task needs its guard; unknown or tiny cubes take
        // the rolled loop. `constant()` bridges the folded total back to host data — a whole smem
        // stage's shape is static, so it always folds.
        let units = comptime!(self.access.stage.units);
        let total_c = total.constant();
        let straight =
            comptime!(matches!(total_c, Some(t) if units > 0 && (t as usize).div_ceil(units) <= 8));
        let d = self.lines_storage_mut::<I2, WP2>();
        if comptime!(straight) {
            let tasks = comptime!((total_c.unwrap() as usize).div_ceil(units));
            #[unroll]
            for t in 0..tasks {
                let i = UNIT_POS as usize + comptime!(t * units);
                if comptime!((t + 1) * units > total_c.unwrap() as usize) {
                    if i < total {
                        d[i] = s.read(physical_coord(
                            i,
                            shape.clone(),
                            start_axis,
                            num_tiled,
                            levels,
                        ));
                    }
                } else {
                    d[i] = s.read(physical_coord(
                        i,
                        shape.clone(),
                        start_axis,
                        num_tiled,
                        levels,
                    ));
                }
            }
        } else {
            let workers = CUBE_DIM as usize;
            let mut i = UNIT_POS as usize;
            while i < total {
                d[i] = s.read(physical_coord(
                    i,
                    shape.clone(),
                    start_axis,
                    num_tiled,
                    levels,
                ));
                i += workers;
            }
        }
    }

    /// Refill this quantized stage's scales side-channel from `src` for the current region: one f32
    /// per block of the sub-tile, from `src`'s windowed scales into the compact self-relative grid
    /// [`smem_quant_info`] laid out. Cooperative across the cube (one block per task, cyclic). The
    /// destination index is the flat block index itself (the grid is row-major, so the per-axis
    /// decode inverts exactly); the source index dots the block coords with `src`'s scale strides,
    /// whose `window_start` already carries the region's base block.
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
            dst_scales[bl] = src_scales[src_idx.fcast::<usize>()];
            bl += workers;
        }
    }

    /// Zero this window: whole lines at the store's width; a checked window skips
    /// cells past the logical bound.
    pub(crate) fn zero(&mut self) {
        let size!(W) = comptime!(self.store.vector_size);
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        for i in 0..total {
            d.write(i, Vector::<T, W>::cast_from(T::from_int(0)));
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

    /// How this store's stages are laid out and filled, carried from the operand's [`Storage`].
    pub(crate) fn stage(&self) -> comptime_type!(StagePlan) {
        comptime!(self.access.stage)
    }

    /// Comptime quant dispatch for a leaf read, mirroring [`fill_from`](MemData::fill_from)'s
    /// storage-element choice: `0` = plain (serve `T` directly); `1` = native (one storage element
    /// per value, `i8`); `>1` = packed `u32`, the packing factor (values per word). The physical
    /// line narrows the served line by exactly this factor.
    // The `let`-then-return is load-bearing: a bare `#[comptime] match` as the method body does not
    // generate the `#[cube]` expand (the value must bind first).
    #[allow(clippy::let_and_return)]
    pub(crate) fn quant_pack(&self) -> comptime_type!(usize) {
        let pack = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => comptime!(info.scheme.num_quants()),
            ComptimeOption::None => 0usize,
        };
        pack
    }

    /// This buffer's byte length (its length is in native lines, so widened by the physical width):
    /// the transaction count a TMA fill into it lands. `T` / `vector_size` are served-typed, so a
    /// quantized buffer widens by the *storage* element and packed width instead (its line count is
    /// the same, one storage line per served line). Unreachable for quant today (only TMA smem
    /// destinations ask, and a TMA source never stages into a quantized register), but computed
    /// correctly rather than refused.
    pub(crate) fn size_bytes(&self) -> u32 {
        let lines = self.store.buffer.len() as u32;
        #[comptime]
        match &self.store.quant {
            ComptimeOption::None => {
                lines * comptime!(T::type_size() as u32 * self.store.vector_size as u32)
            }
            ComptimeOption::Some(info) => {
                let wp = comptime!(self.store.vector_size / info.scheme.num_quants());
                match comptime!(info.scheme.store) {
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => {
                            lines * comptime!(i8::type_size() as u32 * wp as u32)
                        }
                        other => panic!(
                            "MemData::size_bytes: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    QuantStore::PackedU32(_) => {
                        lines * comptime!(u32::type_size() as u32 * wp as u32)
                    }
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

    /// The buffer re-grouped into `Vector<T, W>` lines, which the line-unit base/window
    /// layouts address. `W` is the width the buffer already has, so the regroup is a
    /// no-op; only the cmma row stride widens back to scalars ([`row_stride`](MemData::row_stride)).
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.store.buffer.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](MemData::lines).
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.store
            .buffer
            .as_vectorized_mut()
            .with_vector_size_mut::<W>()
    }

    /// [`lines`](MemData::lines) with the buffer re-typed to the quantized storage
    /// element `I` it truly holds (see [`QuantInfo`]).
    fn lines_storage<I: Numeric, W: Size>(&self) -> &[Vector<I, W>] {
        let storage = unsafe { self.store.buffer.downcast_unchecked::<I>() };
        storage.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines_storage`](MemData::lines_storage): where a quant stage's
    /// [`fill_straight`](MemData::fill_straight) writes the packed storage words. `I == T` on a
    /// plain copy, a same-type reinterpret.
    fn lines_storage_mut<I: Numeric, W: Size>(&mut self) -> &mut [Vector<I, W>] {
        let storage = unsafe { self.store.buffer.downcast_mut_unchecked::<I>() };
        storage.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// The buffer from this window's origin on: the base a cmma load/store addresses,
    /// rows stepping by the scalar [`row_stride`](MemData::row_stride) (cmma takes a line
    /// slice with a scalar stride). Requires an unmasked store whose window doesn't split
    /// rows across storage tiles.
    pub(crate) fn window_slice(&self) -> &[T] {
        let offset = self.window_offset();
        self.store.buffer.slice(offset, self.store.buffer.len())
    }

    /// The mutable twin of [`window_slice`](MemData::window_slice).
    pub(crate) fn window_slice_mut(&mut self) -> &mut [T] {
        let offset = self.window_offset();
        let end = self.store.buffer.len();
        self.store.buffer.slice_mut(offset, end)
    }

    /// Line offset of the window origin: the accumulated `window_start`. On a tiled
    /// store the window must lie within one storage tile.
    fn window_offset(&self) -> usize {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::window_offset: cmma cannot mask an overhang"
        ));
        self.window_start.fcast::<usize>()
    }

    /// The line offset one `edge` step along logical axis `p` moves: the edge decomposed in the
    /// axis's level radix, dotted with the level strides. `edge` is comptime, so this folds to a
    /// constant and [`at`](MemData::at) pays only the multiply. Exact for the tile-aligned windows
    /// [`window_slice`](MemData::window_slice) admits.
    fn step(&self, #[comptime] p: usize, #[comptime] edge: u32) -> u32 {
        let e = comptime!(edge).runtime();
        if comptime!(p < self.layout.start_axis || self.layout.levels == 0) {
            e.fmul(self.layout.physical_strides.at(p))
        } else {
            // One tiling level (`storage_layout` asserts): a grid and a tile digit.
            let jt = comptime!(self.layout.num_tiled + p);
            let finer = self.layout.physical_shape.at(jt);
            let grid = e.fdiv(finer).fmul(self.layout.physical_strides.at(p));
            let tile = e.frem(finer).fmul(self.layout.physical_strides.at(jt));
            grid.fadd(tile)
        }
    }

    /// Scalar stride between matrix rows: the line-unit physical stride of the leaf
    /// tile's row axis, widened back to scalars; a constant on a static store.
    pub(crate) fn row_stride(&self) -> u32 {
        let rows = comptime!(
            self.layout.start_axis + (self.layout.levels + 1) * self.layout.num_tiled - 2
        );
        self.layout
            .physical_strides
            .at(rows)
            .fmul(comptime!(self.store.vector_size as u32).runtime())
    }

    /// Re-view this buffer through `layout` as a [`MatrixView`], carrying its own `check` flag
    /// so the leaf masks without being asked.
    pub(crate) fn masked<W: Size>(&self, layout: BatchMatrix) -> MatrixView<'_, Vector<T, W>> {
        if comptime!(self.store.quant.is_some()) {
            panic!("Tile::matrix: a quantized tile only serves dequantized reads (Tile::flat)")
        }
        MaskedView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window())
                .view(layout),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// [`masked`](MemData::masked) over the storage element `I` a quantized buffer truly holds
    /// (see [`QuantInfo`]); a [`QuantizedView`] pairs it with the scales to dequantize each 2-D
    /// read. The 2-D twin of [`flat_storage`](MemData::flat_storage).
    pub(crate) fn masked_storage<I: Numeric, W: Size>(
        &self,
        layout: BatchMatrix,
    ) -> MatrixView<'_, Vector<I, W>> {
        MaskedView::new(
            self.lines_storage::<I, W>()
                .view(self.base())
                .view(self.window())
                .view(layout),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// The scales as a 2-D view over this same window: [`ScaleLayout`] resolves a window
    /// coordinate to its block's scale, addressed by the same [`BatchMatrix`] as the values, so
    /// both answer the same `(row, col)`. The 2-D twin of [`flat_scales`](MemData::flat_scales).
    fn masked_scales<'a>(&self, info: &'a QuantInfo, layout: BatchMatrix) -> MatrixView<'a, f32> {
        MaskedView::new(
            info.buffer
                .view(ScaleLayout::new(
                    info.strides.clone(),
                    info.window_start,
                    comptime!(info.block.clone()),
                    comptime!(self.store.vector_size),
                ))
                .view(layout),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// The mutable twin of [`masked`](MemData::masked).
    pub(crate) fn masked_mut<W: Size>(
        &mut self,
        layout: BatchMatrix,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        if comptime!(self.store.quant.is_some()) {
            panic!("Tile::matrix_mut: writing a quantized tile requires requantization")
        }
        let base = self.base();
        let window = self.window();
        let check = comptime!(self.access.overhang.masks());
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
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// [`flat`](MemData::flat) over the storage element `I` a quantized buffer truly holds; the
    /// [`QuantizedView`](crate::QuantizedView) wraps it to dequantize each read.
    pub(crate) fn flat_storage<I: Numeric, W: Size>(&self) -> FlatView<'_, Vector<I, W>> {
        FlatView::new(
            self.lines_storage::<I, W>()
                .view(self.base())
                .view(self.window())
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// The scales as a third [`flat`](MemData::flat) over this same window: [`ScaleLayout`]
    /// resolves a window coordinate to its block's scale, then the values' own [`FlatLayout`]
    /// rides on top, so both views answer the same flat position. Masked like the values, so an
    /// overhang line reads scale `0` rather than off the end of the scales.
    fn flat_scales<'a>(&self, info: &'a QuantInfo) -> FlatView<'a, f32> {
        FlatView::new(
            info.buffer
                .view(ScaleLayout::new(
                    info.strides.clone(),
                    info.window_start,
                    comptime!(info.block.clone()),
                    comptime!(self.store.vector_size),
                ))
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// Quantization-transparent [`flat`](MemData::flat): a plain store serves the bare
    /// `Direct` read, a quantized one re-types to the storage element `I` and pairs it with the
    /// scales over the same window, dequantizing each read into `T`. `#[comptime]`, so the plain
    /// path pays nothing.
    pub(crate) fn flat_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
    ) -> TileView<'_, T, I, WP, W, Coords1d> {
        #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => TileView::new_Quantized(QuantizedView::new(
                // The storage view groups at the *physical* width: a packed buffer holds
                // `W / num_quants` elements per served line.
                self.flat_storage::<I, WP>(),
                self.flat_scales(info),
                comptime!(info.scheme),
            )),
            ComptimeOption::None => TileView::new_Direct(self.flat::<W>()),
        }
    }

    /// Quantization-transparent [`masked`](MemData::masked): the 2-D twin of
    /// [`flat_transparent`](MemData::flat_transparent). A plain store serves the bare `Direct`
    /// matrix read; a quantized one re-types to the storage element `I`, pairs it with the scales
    /// over the same [`BatchMatrix`], and dequantizes each `(row, col)` read into `T`. This is
    /// what lets a leaf read a quantized operand straight from gmem, or from a stage still in the
    /// stored element, without a dequantize-into-`f32` fill. `#[comptime]`, so plain pays nothing.
    pub(crate) fn matrix_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
        layout: BatchMatrix,
    ) -> TileView<'_, T, I, WP, W, Coords2d> {
        #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => TileView::new_Quantized(QuantizedView::new(
                self.masked_storage::<I, WP>(layout.clone()),
                self.masked_scales(info, layout),
                comptime!(info.scheme),
            )),
            ComptimeOption::None => TileView::new_Direct(self.masked::<W>(layout)),
        }
    }

    /// The mutable twin of [`flat`](MemData::flat).
    pub(crate) fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        if comptime!(self.store.quant.is_some()) {
            panic!("Tile::flat_mut: writing a quantized tile requires requantization")
        }
        let base = self.base();
        let window = self.window();
        let extent = self.window.extent.clone();
        let check = comptime!(self.access.overhang.masks());
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
        let cols = comptime!(space.extent_at(rank - 1) / self.store.vector_size);
        // Leading (batch) extents are width-invariant; the window extent is the view's shape.
        let shape = self.extent();
        let mut batches = CoordsDyn::new();
        #[unroll]
        for p in 0..rank - 2 {
            let weight = shape.fproduct(comptime!(((p + 1)..(rank - 2)).collect::<Vec<_>>()));
            batches.push(i.fcast::<u32>().fdiv(weight).frem(shape.at(p)));
        }
        self.masked_mut::<W>(BatchMatrix::new(batches, rows, cols))
    }

    /// The [`AccumulateView`] over batch matrix `i`: [`matrix_mut`](MemData::matrix_mut) plus the
    /// [`LaneShare`] these cells carry, so a leaf accumulates through it without being told.
    pub(crate) fn matrix_accumulate<W: Size>(
        &mut self,
        i: usize,
        #[comptime] space: Space,
    ) -> AccumulateView<'_, T, W> {
        let lane_share = comptime!(self.lane_share);
        AccumulateView::new(self.matrix_mut::<W>(i, space), lane_share)
    }

    /// Window down to `region`: shift the origin by the region's tile coordinate times
    /// the sub-tile edge, crop each axis to that edge, re-box the same buffer. `bound`
    /// is carried through unchanged, so the leaf masks correctly at any nesting depth.
    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> MemData<T> {
        let mut origin = Coords::<u32>::new();
        let mut extent = Coords::<u32>::new();
        // Per-axis window_start advances, summed below (chained, so constants fold).
        let mut advances = Coords::<u32>::new();

        let last = comptime!(space.rank() - 1);
        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            // The innermost (vectorized) axis's edge is a line count, so `/ width`.
            let edge = comptime!(if p == last {
                space.partitioner().edge(axis) / self.store.vector_size
            } else {
                space.partitioner().edge(axis)
            });
            let index = region.coord(axis);

            origin.push(
                self.window
                    .origin
                    .at(p)
                    .fadd(index.fmul(edge).fcast::<u32>()),
            );
            extent.push(comptime!(edge as u32).runtime());
            advances.push(
                index
                    .fcast::<u32>()
                    .fmul(self.step(p, comptime!(edge as u32))),
            );
        }
        let rank = comptime!(space.rank());
        let start = self
            .window_start
            .fadd(advances.fsum(comptime!((0..rank).collect::<Vec<_>>())));

        // Re-window the scales alongside the values (a no-op for per-tensor's zero strides).
        let quant = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => ComptimeOption::new_Some(info.window(
                &origin,
                rank,
                comptime!(self.store.vector_size),
            )),
            ComptimeOption::None => ComptimeOption::new_None(),
        };

        MemData::<T> {
            store: Store::<T> {
                buffer: unsafe { self.store.buffer.as_boxed_unchecked() },
                vector_size: comptime!(self.store.vector_size),
                quant,
            },
            // The layout addresses the whole buffer and never narrows; only the window moves.
            layout: self.layout.clone(),
            window: Window::new(origin, extent, self.window.bound.clone()),
            window_start: start,
            // The window no longer covers the buffer, so the straight-through fill is off.
            access: comptime!(Access {
                whole: false,
                ..self.access
            }),
            lane_share: comptime!(join_lane_share(self.lane_share, space.lane_share())),
        }
    }
}

/// The operand's logical extent per axis, folded from its physical `[pre…, grid…, tile…]`
/// shape: passthrough axes pass through, each tiled axis multiplies its per-level factors.
/// Reduces to `physical_shape` for an untiled (strided) operand.
#[cube]
fn logical_bound(
    physical_shape: &Coords<u32>,
    #[comptime] start_axis: usize,
    #[comptime] num_tiled: usize,
    #[comptime] levels: usize,
) -> Coords<u32> {
    let mut bound = Coords::<u32>::new();
    #[unroll]
    for i in 0..start_axis {
        bound.push(physical_shape.at(i));
    }
    #[unroll]
    for a in 0..num_tiled {
        bound.push(physical_shape.fproduct(comptime!(
            (0..=levels)
                .map(|l| start_axis + l * num_tiled + a)
                .collect::<Vec<_>>()
        )));
    }
    bound
}

/// The whole-tile window: `origin = 0`, `extent =` the space's per-axis extents. `Space` is
/// conceptual; the innermost (vectorized) axis's extent is a line count, `/ vector_size`.
#[cube]
fn full_window(
    #[comptime] space: Space,
    #[comptime] vector_size: usize,
) -> (Coords<u32>, Coords<u32>) {
    let mut origin = Coords::<u32>::new();
    let mut extent = Coords::<u32>::new();
    let last = comptime!(space.rank() - 1);

    #[unroll]
    for p in 0..space.rank() {
        origin.push(0);
        let e = comptime!(space.extent(space.axis_at(p)));
        extent.push(comptime!((if p == last { e / vector_size } else { e }) as u32).runtime());
    }

    (origin, extent)
}

/// [`full_window`] for the top gmem tile, where an axis may be [`Dynamic`](crate::Extent): such
/// an axis reads its runtime size from `bound` (the folded logical extent) instead of a comptime
/// constant, so the problem shape never specializes the kernel.
#[cube]
fn top_window(
    #[comptime] space: Space,
    bound: &Coords<u32>,
    #[comptime] vector_size: usize,
) -> (Coords<u32>, Coords<u32>) {
    let mut origin = Coords::<u32>::new();
    let mut extent = Coords::<u32>::new();
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
            Extent::Dynamic => bound.at(p),
        };
        extent.push(size);
    }

    (origin, extent)
}

/// The staged scales side-channel for a quantized smem stage: a compact `Shared` buffer holding one
/// f32 per block of the sub-tile (row-major over the block grid), paired with self-relative strides
/// (`window_start = 0`). [`fill_from`](MemData::fill_from) refills its contents per region;
/// [`masked_scales`](MemData::masked_scales) reads it exactly as it reads a gmem operand's scales.
#[cube]
fn smem_quant_info(
    #[comptime] space: Space,
    #[comptime] scheme: QuantScheme,
) -> ComptimeOption<QuantInfo> {
    let rank = comptime!(space.rank());
    let block = comptime!(block_edges(scheme, rank));
    let (nb, strides_c) = comptime!(smem_scale_grid(&space, &block, scheme));
    let count = comptime!(nb.iter().product::<usize>());
    let scales = Shared::<[f32]>::new_slice(comptime!(count));
    let buffer = unsafe {
        scales
            .inner_ref()
            .downcast_unchecked::<f32>()
            .as_boxed_unchecked()
    };
    let mut strides = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        strides.push(comptime!(strides_c[p] as u32).runtime());
    }
    ComptimeOption::new_Some(QuantInfo {
        buffer,
        strides,
        window_start: 0u32,
        block: comptime!(block),
        scale_shape: comptime!(nb),
        scheme: comptime!(scheme),
    })
}

/// [`smem_quant_info`]'s host data: the per-axis distinct-scale count (`nb`) and its row-major
/// suffix-product strides. Per-tensor is the degenerate single scale — every count `1` and every
/// stride `0`, so a read pins index `0`; a block scheme grids `ceil(extent / block)` per axis.
fn smem_scale_grid(
    space: &Space,
    block: &[usize],
    scheme: QuantScheme,
) -> (Vec<usize>, Vec<usize>) {
    let rank = space.rank();
    let per_tensor = matches!(scheme.level, QuantLevel::Tensor);
    let nb: Vec<usize> = (0..rank)
        .map(|p| {
            if per_tensor {
                1
            } else {
                space.extent_at(p).div_ceil(block[p])
            }
        })
        .collect();
    let strides: Vec<usize> = (0..rank)
        .map(|p| {
            if per_tensor {
                0
            } else {
                nb[p + 1..].iter().product::<usize>()
            }
        })
        .collect();
    (nb, strides)
}

/// The smem physical shape/strides over `space`, line-unit like [`Tile::from_tensor`].
/// `levels == 0` is a plain row-major buffer; `levels == 1` is the `[grid…, tile…]` storage
/// tiling at the space's final tile, each tile a contiguous block.
#[cube]
fn storage_layout(
    #[comptime] space: Space,
    #[comptime] vector_size: usize,
    #[comptime] levels: usize,
) -> (Coords<u32>, Coords<u32>) {
    let (shape_c, strides_c) = comptime!(storage_extents(&space, vector_size, levels));

    let mut shape = Coords::<u32>::new();
    let mut strides = Coords::<u32>::new();
    #[unroll]
    for p in 0..comptime!(shape_c.len()) {
        shape.push(comptime!(shape_c[p]));
        strides.push(comptime!(strides_c[p]));
    }

    (shape, strides)
}

/// [`storage_layout`]'s host data: the physical line extents (`[extents…]` flat, or
/// `[grid…, tile…]`) and their row-major suffix-product strides.
fn storage_extents(space: &Space, vector_size: usize, levels: usize) -> (Vec<u32>, Vec<u32>) {
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

    let shape = extents.iter().map(|&e| e as u32).collect();
    let strides = (0..extents.len())
        .map(|p| extents[p + 1..].iter().product::<usize>() as u32)
        .collect();
    (shape, strides)
}

/// The logical coordinate of physical line `i` in a `[grid…, tile…]` store: suffix-
/// stride digit decode, each logical axis folding its level digits back — by constants
/// on a static store.
#[cube]
fn physical_coord(
    i: usize,
    shape: Coords<u32>,
    #[comptime] start_axis: usize,
    #[comptime] num_tiled: usize,
    #[comptime] levels: usize,
) -> CoordsDyn {
    let x = i.fcast::<u32>();
    let mut out = CoordsDyn::new();
    #[unroll]
    for a in 0..comptime!(start_axis + num_tiled) {
        if comptime!(a < start_axis || levels == 0) {
            out.push(line_digit(x, &shape, a));
        } else {
            // One tiling level (`storage_layout` asserts): the grid digit scales by the
            // tile extent, the tile digit rides along.
            let jt = comptime!(num_tiled + a);
            let l = line_digit(x, &shape, a)
                .fmul(shape.at(jt))
                .fadd(line_digit(x, &shape, jt));
            out.push(l);
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

/// In-kernel twin of cubecl's `TiledViewLayout`, which has no in-kernel
/// constructor: splits each logical axis into its `[grid…, tile…]` parts, then dots
/// the physical strides. Folding arithmetic, so a static store (smem) splits and dots
/// by constants, and an untiled one (`levels == 0`) reduces to the plain strided dot.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct GmemLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) start_axis: usize,
    #[cube(comptime)]
    pub(crate) num_tiled: usize,
    #[cube(comptime)]
    pub(crate) levels: usize,
}

#[cube]
impl Layout for GmemLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        // Per-digit terms, summed below (chained, so a static store's dot folds).
        let mut terms = Sequence::<u32>::new();
        #[unroll]
        for i in 0..comptime!(self.start_axis) {
            terms.push(pos[i].fmul(self.physical_strides.at(i)));
        }
        #[unroll]
        for k in 0..comptime!(self.levels + 1) {
            #[unroll]
            for i in 0..comptime!(self.num_tiled) {
                // Strip the finer blocks, then take this block's digit. The grid
                // (k == 0) keeps the full quotient — it has no enclosing tile.
                let j = comptime!(self.start_axis + k * self.num_tiled + i);
                let divisor = self.physical_shape.fproduct(comptime!(
                    ((k + 1)..=self.levels)
                        .map(|f| self.start_axis + f * self.num_tiled + i)
                        .collect::<Vec<_>>()
                ));
                let quot = pos[comptime!(self.start_axis + i)].fdiv(divisor);
                let digit = if comptime!(k > 0) {
                    quot.frem(self.physical_shape.at(j))
                } else {
                    quot
                };
                terms.push(digit.fmul(self.physical_strides.at(j)));
            }
        }
        let n = comptime!(self.start_axis + (self.levels + 1) * self.num_tiled);
        terms
            .fsum(comptime!((0..n).collect::<Vec<_>>()))
            .fcast::<usize>()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        // Each tiled axis collapses its `levels + 1` factors back to their product.
        let mut semantic = CoordsDyn::new();
        #[unroll]
        for i in 0..comptime!(self.start_axis) {
            semantic.push(self.physical_shape.at(i));
        }
        #[unroll]
        for i in 0..comptime!(self.num_tiled) {
            semantic.push(self.physical_shape.fproduct(comptime!(
                (0..=self.levels)
                    .map(|k| self.start_axis + k * self.num_tiled + i)
                    .collect::<Vec<_>>()
            )));
        }
        semantic
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
#[expand(derive(Clone))]
pub struct Window {
    pub(crate) origin: Coords<u32>,
    pub(crate) extent: Coords<u32>,
    /// Absolute logical extent (the valid region). `shape()` stays `extent` (the tile
    /// cell, so loops cover the whole padded tile), but `is_in_bounds` clips against
    /// `bound` so a checked read/write zeroes / skips the overhang.
    pub(crate) bound: Coords<u32>,
}

#[cube]
impl Window {
    pub fn new(origin: Coords<u32>, extent: Coords<u32>, bound: Coords<u32>) -> Self {
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
            out.push(self.origin.at(i).fadd(pos[i]));
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;

        // The cell can overhang the matrix; a position is valid only if its absolute
        // coordinate (`origin + pos`) is within the logical `bound`.
        #[unroll]
        for i in 0..self.bound.len() {
            valid = valid && self.origin.at(i).fadd(pos[i]) < self.bound.at(i);
        }

        valid
    }
}
