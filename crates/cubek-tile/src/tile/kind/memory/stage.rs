//! Deriving a shared-memory stage from an operand: the [`StageForm`] it takes (physical extents
//! plus the two mappings that address them) and the `smem*` constructors that allocate one.

use cubecl::zspace::SmallVec;
use cubecl::{prelude::*, quant::scheme::QuantScheme, std::quant::view::KnownScale};

use crate::*;

/// The byte alignment a TMA-filled stage's shared buffer must have.
const TMA_STAGE_ALIGNMENT: usize = 128;

#[cube]
impl<T: Numeric> MemData<T> {
    /// Cooperatively materialize a coordinate-backed source into this plain, direct scalar memory
    /// tile. Workers write cyclic positions across it, so the caller must ensure every unit in the
    /// cube owns this window: a property of the level's distribution, not of buffer coverage.
    pub(crate) fn fill_procedural(&mut self, src: &ProceduralData<T>, #[comptime] space: Space) {
        comptime!(assert!(
            self.store.packing == Packing::Plain
                && self.projection.is_direct()
                && self.store.vector_size == 1,
            "MemData::fill_procedural: procedural sources require a plain, direct scalar destination"
        ));
        // Read the destination's runtime window rather than the comptime space so direct copies
        // also work when another operand witnesses a Dynamic extent.
        let shape = self.window.extent.clone();
        let mut dst = self.flat_mut::<Const<1>>();
        let total = dst.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let pos = shape.unravel(i.retyped::<u32>());
            // TODO: staging cannot see its consumer, so this masked fill uses ProceduralData's
            // zero fallback, not every reduction's identity (Max on negatives, Min on positives).
            // A reduction-aware contract must carry validity or the consumer's identity here.
            dst.write(
                i,
                Vector::cast_from(src.evaluate_masked(&pos, comptime!(space.clone()))),
            );
            i += workers;
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of `operand`,
    /// laid out as `storage` and, where `width` is stated, served in lines that wide (an axis gmem
    /// could not vectorize still reaches the leaf in lines). Both are the ring's to state.
    ///
    /// The stage takes the element the operand needs staged: the one it *serves* when the load
    /// decodes it ([`DequantAt::Load`], always so for a plain operand), else the one it is *stored*
    /// in ([`smem_stored`](MemData::smem_stored)). The operand carries which, so no caller asks.
    pub fn stage(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
    ) -> Tile<T> {
        match comptime!(storage.clone()) {
            // The lanes are not memory: the plane holds them, at the depth one region of
            // `level` sits, so the regions below the level window them as they window the
            // operand.
            StageStorage::Lanes { reach } => Tile::<T> {
                tile_kind: TileKind::new_Lanes(Lanes::<T>::new(
                    operand,
                    comptime!(level.clone()),
                    comptime!(reach),
                )),
                space: comptime!(level.child(&operand.space)),
                depth: comptime!(operand.depth + 1),
                levels: comptime!(operand.levels.clone()),
            },
            _ => MemData::<T>::stage_memory(operand, level, storage, width),
        }
    }

    /// [`stage`](MemData::stage) in shared memory.
    fn stage_memory(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
    ) -> Tile<T> {
        let dequant_at = operand.dequant_at();
        match comptime!(dequant_at) {
            DequantAt::Load => {
                let space = comptime!(level.child(&operand.space));
                let projection = operand.projection();
                let units = operand.units();
                let source_width = operand.vector_size();
                let vector_size = comptime!(match width {
                    Some(width) => {
                        assert!(
                            source_width == 1,
                            "MemData::stage: a padded stage assembles its lines from scalar \
                             source cells, so the operand it pads must be unvectorized (it is \
                             served {source_width} wide)"
                        );
                        assert!(
                            width > 1,
                            "MemData::stage: a padded stage width must widen the operand's own \
                             1-wide lines (got {width})"
                        );
                        width
                    }
                    None => source_width,
                });
                // A TMA-filled stage's shared buffer must be TMA-aligned (its bulk copy addresses
                // shared memory directly); a copy-filled one needs only the element alignment
                // `smem` gives. TMA operands are always direct, so the gathered arm never needs it.
                let delivery = operand.delivery();
                let alignment = comptime!(match delivery.is_tma() {
                    true => TMA_STAGE_ALIGNMENT,
                    false => 0usize,
                });
                if comptime!(projection.is_direct()) {
                    MemData::smem_aligned(space, vector_size, storage, units, alignment)
                } else {
                    MemData::smem_gathered(
                        space,
                        vector_size,
                        storage,
                        units,
                        projection,
                        &operand.runtime_map(),
                        operand.window_signed(),
                        operand.window_boundaries(),
                    )
                }
            }
            DequantAt::Read => MemData::smem_stored(operand, level, storage),
        }
    }

    /// [`stage`](MemData::stage) in the element the operand is *stored* in rather than the one it
    /// serves, under [`DequantAt::Read`]: a quantized operand keeps its stored form (`i8` or packed
    /// `u32`) plus scales ([`smem_quant`](MemData::smem_quant)), and the leaf dequantizes on read.
    fn smem_stored(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
    ) -> Tile<T> {
        let space = comptime!(level.child(&operand.space));
        let vector_size = operand.vector_size();
        let units = operand.units();
        match &operand.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                #[comptime]
                match &g.store.quant {
                    // No scheme: the words as they lie where the operand is packed, which is
                    // the only stored form a scheme-less operand has, else a plain stage.
                    ComptimeOption::None => match comptime!(g.store.packing) {
                        Packing::Plain => MemData::smem(space, vector_size, storage, units),
                        packing => {
                            MemData::smem_packed(space, vector_size, storage, units, packing)
                        }
                    },
                    // The store was allocated at the scheme's own storage ([`scheme_packing`]).
                    ComptimeOption::Some(info) => match comptime!(g.store.packing) {
                        Packing::Native => MemData::smem_quant::<i8>(
                            space,
                            vector_size,
                            storage,
                            units,
                            info.table.clone(),
                            comptime!(info.scheme),
                        ),
                        Packing::Packed { field: _ } => MemData::smem_quant::<u32>(
                            space,
                            vector_size,
                            storage,
                            units,
                            info.table.clone(),
                            comptime!(info.scheme),
                        ),
                        Packing::Plain => {
                            panic!("MemData::smem_stored: a quantized store is never plain")
                        }
                    },
                }
            }
            // A tma source has no stored form to keep: it carries no scheme (`quantized` is a
            // strided-builder knob, and a tma tile is scalar), so served == stored. Giving it
            // one must not reuse this arm; see `Staging::new`, which refuses that combination.
            TileKind::TmaGmem(_) => MemData::smem(space, vector_size, storage, units),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("MemData::smem_stored: a fragment is not a stage source")
            }
            TileKind::Procedural(_) | TileKind::Lanes(_) => {
                panic!(
                    "MemData::smem_stored: a procedural tile and the plane's lanes are not a stage source"
                )
            }
        }
    }

    /// Allocate a shared-memory tile over `space`, at physical `vector_size` (allocated natively
    /// wide, then scalar-erased). A `Tiled` stage lays one contiguous block per fragment so a cmma
    /// transaction reads it unstrided; `Strided`, and a final-space stage, is plain row-major.
    ///
    /// `units` is the launch's cube size, `0` when unknown.
    pub fn smem(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
    ) -> Tile<T> {
        MemData::smem_aligned(space, vector_size, storage, units, comptime!(0usize))
    }

    /// [`smem`](MemData::smem) with a minimum byte alignment on the shared
    /// buffer. A TMA-filled stage needs one ([`TMA_STAGE_ALIGNMENT`]); `0`
    /// leaves the buffer at its element alignment.
    pub fn smem_aligned(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] alignment: usize,
    ) -> Tile<T> {
        let form = comptime!(StageForm::dense(&space, vector_size, storage));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        MemData::smem_with_form(
            space,
            vector_size,
            units,
            form,
            map,
            ComptimeOption::new_None(),
            alignment,
        )
    }

    /// [`smem`](MemData::smem) for a *gathered* operand: the stage holds the physical window its
    /// sub-tile reads, compacted ([`Compaction`]), rather than the logical tile, which would
    /// replicate each physical cell by roughly the tap count; the window holds each one once.
    ///
    /// The stage therefore keeps the operand's own [`Projection`] (with the compaction's lattice
    /// quotiented out) instead of becoming direct, so [`Tile::nd`] and [`at`](MemData::at) address
    /// it exactly as they address gmem, and the fill stays a plain box copy.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn smem_gathered(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] projection: Projection,
        map: &RuntimeMap,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    ) -> Tile<T> {
        let form = comptime!(StageForm::gathered(
            &space,
            vector_size,
            storage,
            &projection
        ));
        let stage_map = RuntimeMap {
            coefficients: map.coefficients.clone(),
            residues: Coords::constant(comptime!(vec![0; form.projection.physical_rank()])),
        };
        let stage_map =
            if comptime!(form.projection.is_rational() || form.projection.has_dynamic_scales()) {
                stage_map.stored()
            } else {
                stage_map
            };
        let source = MemData::<T>::pending_source_window(
            comptime!(form.steps.clone()),
            comptime!(signed),
            comptime!(boundaries),
        );
        // A gathered stage is never a TMA destination (TMA operands are
        // direct), so it takes no extra alignment.
        MemData::smem_with_form(
            space,
            vector_size,
            units,
            form,
            stage_map,
            ComptimeOption::new_Some(source),
            comptime!(0usize),
        )
    }

    /// The body every plain smem constructor shares, taking the buffer's [`StageForm`] directly.
    /// `alignment` is the shared buffer's minimum byte alignment (`0` = the element's own).
    #[allow(clippy::too_many_arguments)]
    fn smem_with_form(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] units: usize,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
        #[comptime] alignment: usize,
    ) -> Tile<T> {
        let size!(W) = vector_size;
        let smem = if comptime!(alignment > 0) {
            Shared::<[Vector<T, W>]>::new_aligned_slice(comptime!(form.cells()), alignment)
        } else {
            Shared::<[Vector<T, W>]>::new_slice(comptime!(form.cells()))
        };
        MemData::smem_over(
            space,
            vector_size,
            units,
            &smem,
            ComptimeOption::new_None(),
            comptime!(Packing::Plain),
            form,
            map,
            source,
        )
    }

    /// [`smem`](MemData::smem) over the words a [`packed`](Packing::Packed) operand is stored in:
    /// the line narrows by the packing's factor and the buffer keeps `packing`, so every read
    /// unpacks as one of the global window does. No scales: those are an operand of their own.
    pub(crate) fn smem_packed(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] packing: Packing,
    ) -> Tile<T> {
        let form = comptime!(StageForm::dense(&space, vector_size, storage));
        let size!(WP) = comptime!(packing.physical(vector_size));
        let smem = Shared::<[Vector<u32, WP>]>::new_slice(comptime!(form.cells()));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        MemData::smem_over(
            space,
            vector_size,
            units,
            &smem,
            ComptimeOption::new_None(),
            comptime!(packing),
            form,
            map,
            ComptimeOption::new_None(),
        )
    }

    /// [`smem`](MemData::smem) staging the element `I` an operand is *stored* in (`i8`, or packed
    /// `u32`) rather than the one it serves: the line narrows to `vector_size / pack`, so the stage
    /// is that much smaller and the leaf dequantizes at read, not the fill inflating to `T`.
    ///
    /// Carries a compact `Shared` scales buffer beside the values: one f32 per block of the
    /// sub-tile, refilled per region by [`fill_from`](MemData::fill_from).
    pub(crate) fn smem_quant<I: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        table: ComptimeOption<Box<[f32]>>,
        #[comptime] scheme: QuantScheme,
    ) -> Tile<T> {
        // One stored line is one served line, just narrower, so only the element and width change:
        // the layout and window below are the same grid either way.
        let form = comptime!(StageForm::dense(&space, vector_size, storage));
        let size!(WP) = comptime!(vector_size / scheme.num_quants());
        let smem = Shared::<[Vector<I, WP>]>::new_slice(comptime!(form.cells()));
        let quant = smem_quant_info(comptime!(space.clone()), table, comptime!(scheme));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        MemData::smem_over(
            space,
            vector_size,
            units,
            &smem,
            quant,
            comptime!(scheme_packing(scheme)),
            form,
            map,
            ComptimeOption::new_None(),
        )
    }

    /// The body every smem constructor shares, taking the allocated slice (so the element is the
    /// caller's) and the buffer's [`StageForm`]: scalar-erases the slice to the served `T` (views
    /// recover it via [`lines_storage`](MemData::lines_storage)) and windows the whole buffer.
    ///
    /// `units` is the launch's cube size, `0` when unknown ([`Access::units`](super::Access)).
    #[allow(clippy::too_many_arguments)]
    fn smem_over<S: CubePrimitive>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] units: usize,
        smem: &Shared<[S]>,
        quant: ComptimeOption<QuantInfo>,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
    ) -> Tile<T> {
        // Shared memory is always an address: a stage is read back by the
        // instruction that consumes it, which is the one thing a sink cannot do.
        let backing = Backing::<T>::new_Buffer(unsafe {
            smem.inner_ref()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        });
        let (physical_shape, physical_strides) = storage_layout(comptime!(form.clone()));
        let (origin, extent) = full_window(comptime!(form.clone()));
        // Smem never overhangs its own buffer, so the bound is the extent and checks are off.
        let bound = extent.clone();
        let gmem_projection = comptime!(form.positional.clone());
        Tile::<T> {
            tile_kind: TileKind::new_Smem(MemData::<T> {
                store: Store::<T> {
                    backing,
                    vector_size,
                    quant,
                    packing: comptime!(packing),
                },
                layout: BufferLayout {
                    physical_shape,
                    physical_strides,
                    projection: gmem_projection,
                },
                // Stage origins are never negative and smem never overhangs (`Overhang::Never`
                // below), so the boundary policy is never consulted.
                //
                // Empty is the only list it *can* be: this window is shaped over the buffer's own
                // dims (a tiled stage's fragments), its sub-windows over coordinates, so a per-axis
                // list minted here would land on the wrong axes one level down.
                window: Window::new(origin, extent, bound, false, comptime!(SmallVec::new())),
                projection: comptime!(form.projection),
                map,
                offsets: Coords::<i32>::new(),
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang: Overhang::Never,
                    write: Write::Replace,
                    units,
                    // A stage is allocated here, whole: one storage tile over the buffer.
                    storage: Storage::Strided,
                }),
                lanes: comptime!(LaneShare::Repeated),
                split_share: comptime!(SplitShare::Whole),
                init_from: comptime!(InitFrom::Cell),
                source_window: source,
                lands: false,
            }),
            space: comptime!(space),
            depth: comptime!(0usize),
            levels: comptime!(Vec::new()),
        }
    }

    /// One plane's landing: a dense, scalar stage over `space`, one window per plane of the
    /// cube in one shared buffer, this plane's found by the walk's own decode of the hardware
    /// position. The buffer comes back beside the tile, for the lanes that fill it.
    pub(crate) fn landing(
        #[comptime] space: Space,
        #[comptime] units: usize,
        #[comptime] planes: usize,
    ) -> (Tile<T>, Shared<[T]>) {
        let cells = comptime!(
            (0..space.rank())
                .map(|p| space.extent_at(p))
                .product::<usize>()
        );
        let start = Takers::position(Takers::Planes) * cells;
        let end = start + cells;
        let window =
            Shared::<[T]>::new_slice(comptime!(cells * planes)).map(|all| &all[start..end]);
        let form = comptime!(StageForm::dense(&space, 1, StageStorage::Strided));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let tile = MemData::smem_over(
            space,
            1usize,
            units,
            &window,
            ComptimeOption::new_None(),
            comptime!(Packing::Plain),
            form,
            map,
            ComptimeOption::new_None(),
        );
        (tile, window)
    }

    /// An unfilled [`SourceWindow`] for a gathered stage: the comptime geometry is the stage's own,
    /// while the origin and bound are written by each [`fill_from`](MemData::fill_from) from the
    /// operand that fill reads.
    fn pending_source_window(
        #[comptime] steps: SmallVec<[usize; Space::MAX_RANK]>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    ) -> SourceWindow {
        let rank = comptime!(steps.len());
        let mut origin = Coords::<i32>::new();
        let mut bound = Coords::<u32>::new();
        #[unroll]
        for _ in 0..rank {
            origin.push(0i32);
            bound.push(0u32);
        }
        SourceWindow {
            origin,
            bound,
            steps: comptime!(steps),
            signed: comptime!(signed),
            boundaries: comptime!(boundaries),
        }
    }
}

/// The whole-buffer window of a stage: `origin = 0`, `extent =` its own physical extents.
#[cube]
fn full_window(#[comptime] form: StageForm) -> (Coords<i32>, Coords<u32>) {
    let mut origin = Coords::<i32>::new();
    let mut extent = Coords::<u32>::new();

    #[unroll]
    for p in 0..comptime!(form.extents.len()) {
        origin.push(0);
        extent.push(comptime!(form.extents[p] as u32).runtime());
    }

    (origin, extent)
}

/// The staged scales side-channel for a quantized smem stage: a compact `Shared` buffer, one f32
/// per block of the sub-tile, row-major, self-relative (`window_start = 0`), refilled per region
/// by [`fill_from`](MemData::fill_from), read by [`transparent`](MemData::transparent) as gmem's.
#[cube]
fn smem_quant_info(
    #[comptime] space: Space,
    table: ComptimeOption<Box<[f32]>>,
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
    #[allow(clippy::needless_range_loop)]
    #[unroll]
    for p in 0..rank {
        strides.push(comptime!(strides_c[p] as u32).runtime());
    }
    ComptimeOption::new_Some(QuantInfo {
        buffer,
        // The fill folds a two-level source's global level into the staged grid
        // ([`MemData::stage_scales`]), so the stage serves effective scales under the one-level
        // form of the scheme; keeping the two-level level here would fail cubecl's binding check.
        known: KnownScale::new_None(),
        strides,
        window_start: 0u32,
        block: comptime!(block),
        extent: comptime!(window_extents(&space, rank)),
        // A stage only keeps its quantized form when the read is what decodes it; that is the one
        // path reaching here.
        dequant_at: comptime!(DequantAt::Read),
        scale_shape: comptime!(nb),
        // The gmem table rides through: it is never staged, only re-read.
        table,
        scheme: comptime!(staged_scheme(scheme)),
    })
}

/// [`smem_quant_info`]'s host data: the per-axis distinct-scale count (`nb`) and its row-major
/// suffix-product strides. Per-tensor is the degenerate single scale; every count `1` and every
/// stride `0`, so a read pins index `0`; a block scheme grids `ceil(extent / block)` per axis.
fn smem_scale_grid(
    space: &Space,
    block: &[usize],
    scheme: QuantScheme,
) -> (Vec<usize>, Vec<usize>) {
    let rank = space.rank();
    let per_tensor = scheme.block_size().is_none();
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

/// How a gathered fill's two sides relate: the [`Compaction`] of the source's own map, which the
/// destination must be addressed by for its physical box to be the source's window. `None` when
/// neither side gathers: nothing to compact, and no extent read (a top-level `Dynamic` has none).
///
/// `space` is the destination's, sizing the source's window; `vector_size` the destination stage's
/// served width, for [`Compaction::of`]. The assert pins the two *mappings* together, which
/// a `Tile::copy_from` caller can get wrong; sizes are [`fill_straight`](MemData::fill_straight)'s.
pub(crate) fn stage_compaction(
    src: &Projection,
    dst: &Projection,
    vector_size: usize,
    space: &Space,
) -> Option<Compaction> {
    // A `MemData` carries the *coordinate*-space map ([`Projection::untiled`]), where storage
    // tiling has already folded back into the one coordinate its fragments are digits of. Direct
    // there is exactly "no gather", so a tiled buffer takes this early return like any other.
    if src.is_direct() && dst.is_direct() {
        return None;
    }
    // A partition is not a gather: nothing aliases and every window is a box, so its stage is
    // the dense copy of the logical tile a direct operand's is, and the fill reads the source
    // box straight through its own digits.
    if src.composition() == Composition::Disjoint && dst.is_direct() {
        return None;
    }
    let compaction = Compaction::new(src, vector_size, |axis| space.extent(axis));
    assert!(
        compaction.projection() == dst,
        "stage_compaction: a gathered source fills the compacted stage of its own \
         projection, addressed by {:?}, but the destination is addressed by {dst:?}",
        compaction.projection()
    );
    Some(compaction)
}

/// A stage's buffer: the physical extents it takes and the two mappings that address them. The one
/// place a dense stage and a gathered one differ, so [`smem_over`](MemData::smem_over) builds
/// either without knowing which it is.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct StageForm {
    /// Physical extents in lines, innermost already divided by the store width.
    extents: Vec<usize>,
    /// The buffer's own per-position map, what [`BufferLayout`] splits coordinates through.
    positional: Projection,
    /// How the staged tile's logical axes address those extents.
    projection: Projection,
    /// What a stage coordinate is multiplied by to land on the source, per physical axis. All `1`
    /// for a dense stage, which is a copy of the tile and shares its coordinates.
    steps: SmallVec<[usize; Space::MAX_RANK]>,
}

impl StageForm {
    /// A materialized dense copy of the logical tile: what every direct operand stages into. An
    /// empty `nesting` is a plain row-major buffer; each block in it adds a `[grid…, block…]`
    /// split, so the buffer lays the innermost block down contiguously.
    fn dense(space: &Space, vector_size: usize, stage: StageStorage) -> StageForm {
        let nesting = stage.nesting(space);
        StageForm {
            extents: StageForm::dense_extents(space, vector_size, &nesting),
            positional: Projection::of_tiling(StorageTiling::uniform(space.rank(), nesting.len())),
            // A dense stage is a copy of the tile itself, so it addresses its own buffer directly
            // whatever the operand it stages was gathered through.
            projection: Projection::direct_over(space),
            steps: SmallVec::new(),
        }
    }

    /// The compacted window a gathered operand stages into ([`Compaction`]): one cell per element
    /// its sub-tile reads, addressed by the operand's own map with the lattice quotiented out.
    /// Always row-major: an affine map cannot also be storage-tiled ([`Projection::validate`]).
    fn gathered(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        projection: &Projection,
    ) -> StageForm {
        // `Tiled` comes from a cmma leaf, which `Staging::new` refuses a gathered operand for. The
        // nesting has nowhere to go here, so it is refused rather than silently dropped.
        assert!(
            matches!(stage, StageStorage::Strided),
            "StageForm: a gathered operand stages into a plain row-major window, but {stage:?} \
             storage was asked for"
        );
        let compaction = Compaction::new(projection, vector_size, |axis| space.extent(axis));
        let extents = compaction.line_extents(vector_size);
        StageForm {
            positional: Projection::of_tiling(StorageTiling::uniform(extents.len(), 0)),
            projection: compaction.projection().clone(),
            steps: compaction.steps().iter().copied().collect(),
            extents,
        }
    }

    /// How many lines the buffer holds.
    fn cells(&self) -> usize {
        self.extents.iter().product()
    }

    /// Row-major suffix-product strides over [`extents`](StageForm::extents).
    fn strides(&self) -> Vec<usize> {
        (0..self.extents.len())
            .map(|p| self.extents[p + 1..].iter().product())
            .collect()
    }

    /// A dense stage's physical line extents: `[extents…]` flat, or `[grid…, …, block…]`, one grid
    /// per level of `nesting`. A level contributes how many of the next block down it holds; the
    /// innermost contributes its own extents.
    fn dense_extents(space: &Space, vector_size: usize, nesting: &[Space]) -> Vec<usize> {
        let rank = space.rank();
        let mut extents = Vec::new();
        let mut outer = space;
        for block in nesting {
            for p in 0..rank {
                let (e, b) = (outer.extent_at(p), block.extent_at(p));
                assert!(
                    e.is_multiple_of(b),
                    "MemData::smem: a {b}-element storage block must divide the {e}-element block \
                     enclosing it on axis {p}"
                );
                extents.push(e / b);
            }
            outer = block;
        }
        for p in 0..rank {
            extents.push(outer.extent_at(p));
        }
        // Rounded up, not truncated: a padded stage's innermost extent need not fill whole lines,
        // and the spare lanes of the last one are its padding. `fill_extent` refuses the case where
        // the rounding would mean the stage and its source disagree; every fill path asks it.
        let last = extents.len() - 1;
        extents[last] = extents[last].div_ceil(vector_size);
        extents
    }
}

/// A stage's physical shape and strides, in lines like [`Tile::of`]'s.
#[cube]
fn storage_layout(#[comptime] form: StageForm) -> (Coords<u32>, Coords<u32>) {
    let strides_c = comptime!(form.strides());

    let mut shape = Coords::<u32>::new();
    let mut strides = Coords::<u32>::new();
    #[allow(clippy::needless_range_loop)]
    #[unroll]
    for p in 0..comptime!(form.extents.len()) {
        shape.push(comptime!(form.extents[p] as u32));
        strides.push(comptime!(strides_c[p] as u32));
    }

    (shape, strides)
}

/// What a padded fill needs beyond the two boxes: `width` scalar source cells assembled per
/// destination line, and `lanes` the innermost extent past which those cells are padding; `None`
/// for a `Dynamic` extent, where the source's own bounds check zeroes them ([`fill_extent`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    pub(crate) width: usize,
    pub(crate) lanes: Option<usize>,
    /// The physical rank both boxes share, which only this path needs: the 1:1 copy reads its
    /// line whole and never rebuilds a coordinate.
    pub(crate) rank: usize,
}

impl StageStorage {
    /// The storage-tiling nesting a stage over `space` gets: the blocks its buffer lays down
    /// contiguously, coarse to fine, each dividing the one before it (`space` is the implicit
    /// outermost). Empty is a plain row-major buffer.
    ///
    /// A `Tiled` stage groups the stated block, the fragment a cmma transaction reads unstrided,
    /// projected onto the operand's own axes. A space that is the block already has no grid left to
    /// tile, so it stays plain whatever the layout asks for.
    pub(crate) fn nesting(&self, space: &Space) -> Vec<Space> {
        match self {
            StageStorage::Lanes { .. } => {
                panic!("StageStorage::Lanes: the plane's lanes are not shared memory")
            }
            StageStorage::Tiled { block } => {
                let nested = Space::new(
                    &space
                        .axes()
                        .map(|axis| {
                            let edge = block
                                .iter()
                                .find(|&&(a, _)| a == axis)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "StageStorage::Tiled: the block states no edge for {axis:?}"
                                    )
                                })
                                .1;
                            (axis, edge)
                        })
                        .collect::<Vec<_>>(),
                );
                if &nested == space {
                    return Vec::new();
                }
                vec![nested]
            }
            StageStorage::Strided => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// `16 -> 8 -> 4` on both axes, so the space, its first child and its leaf are three
    /// distinct block shapes to nest.
    fn space() -> (Space, Vec<Level>) {
        (
            Space::new(&[(M, 16), (N, 16)]),
            Levels::leaf(&[(M, 4), (N, 4)])
                .walk(&[(M, 2), (N, 2)])
                .walk_every(&[M, N])
                .build(),
        )
    }

    /// [`space`] plus the ungathered innermost axis a gathered projection is required to carry,
    /// cut once to `8 x 8 x 4`.
    fn gathered_space() -> Space {
        Space::new(&[(M, 8), (N, 8), (K, 4)])
    }

    /// No nesting is the plain row-major buffer: the space's own extents, innermost in lines.
    #[test]
    fn flat_nesting_is_the_space_itself() {
        let (space, _) = space();
        assert_eq!(StageForm::dense_extents(&space, 1, &[]), vec![16, 16]);
        assert_eq!(StageForm::dense_extents(&space, 4, &[]), vec![16, 4]);
    }

    /// One block is the `[grid…, tile…]` split: each axis holds `16 / 4` tiles of `4`.
    #[test]
    fn one_block_splits_grid_and_tile() {
        let (space, levels) = space();
        assert_eq!(
            StageForm::dense_extents(&space, 1, &[space.leaf(&levels)]),
            vec![4, 4, 4, 4]
        );
    }

    /// Two nested blocks add a middle grid, each level counting how many of the block below it
    /// it holds: `16 = 2 x (8 = 2 x (4))`.
    #[test]
    fn two_blocks_nest() {
        let (space, levels) = space();
        let nesting = [levels[0].child(&space), space.leaf(&levels)];
        let extents = StageForm::dense_extents(&space, 1, &nesting);
        assert_eq!(extents, vec![2, 2, 2, 2, 4, 4]);
        // The nesting only regroups the buffer, never resizes it.
        assert_eq!(extents.iter().product::<usize>(), space.cells());
    }

    /// The strides a buffer's extents imply, row-major: `[grid…, tile…]` or plain, the same rule.
    #[test]
    fn a_form_strides_row_major() {
        let (space, levels) = space();
        let form = StageForm::dense(&space, 4, StageStorage::Strided);
        assert_eq!(form.extents, vec![16, 4]);
        assert_eq!(form.strides(), vec![4, 1]);
        assert_eq!(form.cells(), space.cells() / 4);

        let tiled = StageForm::dense(
            &space,
            1,
            StageStorage::Tiled {
                block: space.leaf(&levels).extents(),
            },
        );
        assert_eq!(tiled.extents, vec![4, 4, 4, 4]);
        assert_eq!(tiled.strides(), vec![64, 16, 4, 1]);
    }

    /// A gathered stage is the compacted window, not the logical tile: `M` and `N` here map onto
    /// one physical axis, so the stage holds their receptive field instead of their product. `K`
    /// rides identity innermost, as every gathered projection must ([`Projection::validate`]).
    #[test]
    fn a_gathered_form_is_the_compacted_window() {
        let space = gathered_space();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        let form = StageForm::gathered(&space, 1, StageStorage::Strided, &projection);
        // 8 x 8 logical cells over 1 + 7 + 7 physical ones, times the ungathered 4 of `K`.
        assert_eq!(form.extents, vec![15, 4]);
        assert_eq!(form.cells(), 60);
        assert_eq!(form.projection, projection);
        assert!(form.positional.is_direct());
    }

    /// A gathered operand's stage is plain row-major; a storage-tiled one has nowhere to nest.
    #[test]
    #[should_panic(expected = "plain row-major window")]
    fn a_gathered_form_refuses_tiled_storage() {
        let space = gathered_space();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        StageForm::gathered(
            &space,
            1,
            StageStorage::Tiled {
                block: vec![(M, 4), (N, 4), (K, 4)],
            },
            &projection,
        );
    }

    /// A block that does not divide the one enclosing it has no `[grid…, block…]` split.
    #[test]
    #[should_panic(expected = "must divide")]
    fn a_block_must_divide_its_enclosing_block() {
        let (space, levels) = space();
        // Reversed: the coarse block sits inside the fine one.
        StageForm::dense_extents(&space, 1, &[space.leaf(&levels), levels[0].child(&space)]);
    }

    /// A `Tiled` stage groups the stated block; a space that is the block already has no grid
    /// left, so it stays plain.
    #[test]
    fn the_nesting_follows_the_layout() {
        let (space, levels) = space();
        let tiled = StageStorage::Tiled {
            block: space.leaf(&levels).extents(),
        };
        assert!(tiled.nesting(&space)[0] == space.leaf(&levels));
        assert!(StageStorage::Strided.nesting(&space).is_empty());
        assert!(tiled.nesting(&space.leaf(&levels)).is_empty());
    }
}
