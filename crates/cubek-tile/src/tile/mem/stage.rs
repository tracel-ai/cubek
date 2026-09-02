//! Deriving a shared-memory stage from an operand: the [`StageForm`] it takes (physical extents
//! plus the two mappings that address them) and the `smem*` constructors that allocate one.

use cubecl::zspace::SmallVec;
use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue},
    std::quant::view::KnownScale,
};

use crate::*;

/// Comptime metadata bundled when constructing a shared-memory stage.
#[derive(Clone)]
pub(crate) struct StageMeta {
    pub space: Space,
    pub vector_size: usize,
    pub stage: StagePlan,
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// Cooperatively materialize a coordinate-backed source into this plain, direct scalar memory
    /// tile. The caller must ensure that every unit in the cube owns this destination window:
    /// workers write cyclic positions across it. That is a property of the level's distribution,
    /// rather than whether this window covers its backing buffer.
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
            let pos = unravel(&shape, i.fcast::<u32>());
            // TODO: Staging has no knowledge of its eventual consumer, so this generic masked
            // fill uses ProceduralData's zero fallback. That is not the identity for every
            // reduction (notably Max over negative values and Min over positive values). A
            // reduction-aware staging contract must carry either validity or the consumer's
            // identity into this fill before staged procedural reductions can be fully correct.
            dst.write(
                i,
                Vector::cast_from(src.evaluate_masked(&pos, comptime!(space.clone()))),
            );
            i += workers;
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of `operand`, in
    /// the element that operand needs staged: the one it *serves* when the load is what decodes it
    /// ([`DequantAt::Load`], and always for a plain operand, whose served and stored elements are the
    /// same), else the one it is *stored* in ([`smem_stored`](MemData::smem_stored)). The operand
    /// carries which, so a caller stages it without asking whether it is quantized at all.
    pub fn smem_like(operand: &Tile<T>) -> Tile<T> {
        let dequant_at = operand.dequant_at();
        match comptime!(dequant_at) {
            DequantAt::Load => {
                let space = comptime!(operand.space.divide());
                let projection = operand.projection();
                // The stage is one level down, so it takes the operand's plan from the next level
                // on: its own residence was consumed by the decision to build it.
                let source_plan = operand.stage_plan();
                let stage = comptime!(source_plan.descend());
                // A padded stage is served wider than the operand it is filled from: the buffer
                // owns its own layout, so an axis global memory could not vectorize still reaches
                // the leaf in lines. `fill_straight` widens across the two.
                let source_width = operand.vector_size();
                let vector_size = comptime!(source_plan.effective_width(source_width));

                if comptime!(projection.is_direct()) {
                    MemData::smem(space, vector_size, stage)
                } else {
                    MemData::smem_gathered(
                        space,
                        vector_size,
                        stage,
                        projection,
                        &operand.runtime_map(),
                        operand.window_signed(),
                        operand.window_boundaries(),
                    )
                }
            }
            DequantAt::Read => MemData::smem_stored(operand),
        }
    }

    /// [`smem_like`](MemData::smem_like) in the element the operand is *stored* in rather than the
    /// one it serves: a quantized operand keeps its stored form (native `i8`, or `u32` words when
    /// the scheme packs several values each) and its scales, so the leaf dequantizes at the read
    /// (see [`smem_quant`](MemData::smem_quant)) instead of the fill inflating the stage to `T`.
    /// Reached only through [`smem_like`](MemData::smem_like), on an operand whose quantized form
    /// runs [`DequantAt::Read`].
    fn smem_stored(operand: &Tile<T>) -> Tile<T> {
        let space = comptime!(operand.space.divide());
        let vector_size = operand.vector_size();
        let source_plan = operand.stage_plan();
        let stage = comptime!(source_plan.descend());
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
                                info.table.clone(),
                                comptime!(info.scheme),
                            ),
                            other => panic!(
                                "MemData::smem_stored: native quant storage element {:?} is not wired (i8 only)",
                                other
                            ),
                        },
                        QuantStore::PackedU32(_) => MemData::smem_quant::<u32>(
                            space,
                            vector_size,
                            stage,
                            info.table.clone(),
                            comptime!(info.scheme),
                        ),
                        other => panic!(
                            "MemData::smem_stored: quant storage {:?} is not wired (native or packed-u32)",
                            other
                        ),
                    },
                }
            }
            // A tma source has no stored form to keep: it carries no scheme (`quantized` is a
            // strided-builder knob, and a tma tile is scalar), so served == stored. Giving it
            // one must not reuse this arm; see `Staging::new`, which refuses that combination.
            TileKind::TmaGmem(_) => MemData::smem(space, vector_size, stage),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("MemData::smem_stored: a fragment is not a stage source")
            }
            TileKind::Procedural(_) => {
                panic!("MemData::smem_stored: a procedural tile is not a stage source")
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
        let form = comptime!(StageForm::dense(&space, vector_size, stage.storage));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let meta = comptime!(StageMeta {
            space,
            vector_size,
            stage,
        });
        MemData::smem_with_form(meta, form, map, ComptimeOption::new_None())
    }

    /// [`smem`](MemData::smem) for a *gathered* operand: the stage holds the physical window its
    /// sub-tile reads, compacted ([`Compaction`]), rather than the logical tile. Several logical
    /// cells of a gather read the same physical one, so staging the logical tile would replicate
    /// elements by roughly the tap count; staging the window holds each one once.
    ///
    /// The stage therefore keeps the operand's own [`Projection`] (with the compaction's lattice
    /// quotiented out) instead of becoming direct: its buffer is the same shape of window the gmem
    /// operand was, so [`Tile::nd`] and [`at`](MemData::at) address it through exactly the machinery
    /// they address gmem through, and the fill stays a plain box copy.
    pub(crate) fn smem_gathered(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
        #[comptime] projection: Projection,
        map: &RuntimeMap,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
    ) -> Tile<T> {
        let form = comptime!(StageForm::gathered(
            &space,
            vector_size,
            stage.storage,
            &projection
        ));
        let stage_map = RuntimeMap {
            coefficients: map.coefficients.clone(),
            residues: const_coords(comptime!(vec![0; form.projection.physical_rank()])),
        };
        let stage_map =
            if comptime!(form.projection.is_rational() || form.projection.has_dynamic_scales()) {
                stage_map.stored()
            } else {
                stage_map
            };
        let meta = comptime!(StageMeta {
            space,
            vector_size,
            stage,
        });
        let source = MemData::<T>::pending_source_window(
            comptime!(form.steps.clone()),
            comptime!(signed),
            comptime!(boundaries),
        );
        MemData::smem_with_form(meta, form, stage_map, ComptimeOption::new_Some(source))
    }

    /// The body every smem constructor shares, taking the buffer's [`StageForm`] directly.
    fn smem_with_form(
        #[comptime] meta: StageMeta,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
    ) -> Tile<T> {
        let size!(W) = meta.vector_size;
        let smem = Shared::<[Vector<T, W>]>::new_slice(comptime!(form.cells()));
        MemData::smem_over(
            meta,
            &smem,
            ComptimeOption::new_None(),
            comptime!(Packing::Plain),
            form,
            map,
            source,
        )
    }

    /// [`smem`](MemData::smem) staging the element an operand is *stored* in rather than the one it
    /// serves: `I` is that element (`i8`, or `u32` when the scheme packs several values per word)
    /// and the line narrows to `vector_size / pack`, so the stage is that much smaller and a leaf
    /// dequantizes at the read instead of the fill inflating it to `T`. Carries a compact `Shared`
    /// scales buffer beside the values (one f32 per block of the sub-tile, refilled per region by
    /// [`fill_from`](MemData::fill_from)).
    pub(crate) fn smem_quant<I: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
        table: ComptimeOption<Box<[f32]>>,
        #[comptime] scheme: QuantScheme,
    ) -> Tile<T> {
        // One stored line is one served line, just narrower, so only the element and width change:
        // the layout and window below are the same grid either way.
        let form = comptime!(StageForm::dense(&space, vector_size, stage.storage));
        let size!(WP) = comptime!(vector_size / scheme.num_quants());
        let smem = Shared::<[Vector<I, WP>]>::new_slice(comptime!(form.cells()));
        let quant = smem_quant_info(comptime!(space.clone()), table, comptime!(scheme));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let meta = comptime!(StageMeta {
            space,
            vector_size,
            stage,
        });
        MemData::smem_over(
            meta,
            &smem,
            quant,
            comptime!(scheme_packing(scheme)),
            form,
            map,
            ComptimeOption::new_None(),
        )
    }

    /// The body every smem constructor shares: everything but the allocation's element (which is why
    /// it takes the allocated slice rather than making it) and the buffer's [`StageForm`]. Scalar-
    /// erases the slice to the served `T` (the views recover the stored element through
    /// [`lines_storage`](MemData::lines_storage)) and wraps it in the whole-buffer window.
    fn smem_over<S: CubePrimitive>(
        #[comptime] meta: StageMeta,
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
                    vector_size: meta.vector_size,
                    quant,
                    packing: comptime!(packing),
                },
                layout: GmemLayout {
                    physical_shape,
                    physical_strides,
                    projection: gmem_projection,
                },
                // Stage origins are never negative, and smem never overhangs (`Overhang::Never`
                // below), so the boundary policy is never consulted. The empty list is the only
                // thing it *can* be: this window is shaped over the buffer's own dims (a tiled
                // stage's grid and tile fragments), while the sub-windows that inherit it are
                // shaped over coordinates, so any per-axis list minted here would land on the
                // wrong axes one level down.
                window: Window::new(origin, extent, bound, false, comptime!(SmallVec::new())),
                projection: comptime!(form.projection),
                map,
                offsets: Coords::<i32>::new(),
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang: Overhang::Never,
                    write: Write::Replace,
                    stage: meta.stage,
                }),
                lanes: comptime!(Lanes {
                    share: LaneShare::Whole,
                    work: LaneWork::Repeated,
                }),
                split_share: comptime!(SplitShare::Whole),
                init_from: comptime!(InitFrom::Cell),
                source_window: source,
            }),
            space: comptime!(meta.space),
        }
    }

    /// An unfilled [`SourceWindow`] for a gathered stage: the comptime geometry is the stage's own,
    /// while the origin and bound are written by each [`fill_from`](MemData::fill_from) from the
    /// operand that fill reads.
    fn pending_source_window(
        #[comptime] steps: SmallVec<[usize; MAX_AXES]>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
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

/// The staged scales side-channel for a quantized smem stage: a compact `Shared` buffer holding one
/// f32 per block of the sub-tile (row-major over the block grid), paired with self-relative strides
/// (`window_start = 0`). [`fill_from`](MemData::fill_from) refills its contents per region;
/// [`transparent`](MemData::transparent) reads it exactly as it reads a gmem operand's scales.
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
/// destination must be addressed by, since that is what makes its physical box the window the
/// source was windowed down to. `None` when neither side gathers, which is every fill the engine
/// ran before compaction existed: there is nothing to compact, and no extent is read, since a
/// top-level `Dynamic` axis has none to read.
///
/// `space` is the destination's, used here to size the source's window. The assert pins the two
/// *mappings* together, which is what a caller not carrying the same axes on both sides
/// (`Tile::copy_from` is public) gets wrong; the two *sizes* are pinned separately, by
/// [`fill_straight`](MemData::fill_straight) against its own line count. `vector_size` is the
/// destination stage's served width, threaded to [`Compaction::of`].
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
    let compaction = Compaction::of(src, vector_size, |axis| space.extent(axis));
    assert!(
        compaction.projection() == dst,
        "stage_compaction: a gathered source fills the compacted stage of its own \
         projection, addressed by {:?}, but the destination is addressed by {dst:?}",
        compaction.projection()
    );
    Some(compaction)
}

/// A stage's buffer: the physical extents it takes and the two mappings that address them. The one
/// place a dense stage and a gathered one differ, so [`smem_over`](MemData::smem_over) builds either
/// without knowing which it is.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct StageForm {
    /// Physical extents in lines, innermost already divided by the store width.
    extents: Vec<usize>,
    /// The buffer's own per-position map, what [`GmemLayout`] splits coordinates through.
    positional: Projection,
    /// How the staged tile's logical axes address those extents.
    projection: Projection,
    /// What a stage coordinate is multiplied by to land on the source, per physical axis. All `1`
    /// for a dense stage, which is a copy of the tile and shares its coordinates.
    steps: SmallVec<[usize; MAX_AXES]>,
}

impl StageForm {
    /// A materialized dense copy of the logical tile: what every direct operand stages into. An
    /// empty `nesting` is a plain row-major buffer; each block in it adds a `[grid…, block…]` split,
    /// so the buffer lays the innermost block down contiguously.
    fn dense(space: &Space, vector_size: usize, stage: StageStorage) -> StageForm {
        let nesting = stage_nesting(space, stage);
        StageForm {
            extents: storage_extents(space, vector_size, &nesting),
            positional: Projection::of_tiling(StorageTiling::uniform(space.rank(), nesting.len())),
            // A dense stage is a copy of the tile itself, so it addresses its own buffer directly
            // whatever the operand it stages was gathered through.
            projection: Projection::direct_over(space),
            steps: SmallVec::new(),
        }
    }

    /// The compacted window a gathered operand stages into ([`Compaction`]): one cell per element
    /// its sub-tile reads, addressed by the operand's own map with the lattice quotiented out.
    /// Always plain row-major, since an affine map cannot also be storage-tiled
    /// ([`Projection::validate`]).
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
        let compaction = Compaction::of(projection, vector_size, |axis| space.extent(axis));
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

/// The storage-tiling nesting a stage over `space` gets: the blocks its buffer lays down
/// contiguously, coarse to fine, each dividing the one before it (`space` is the implicit
/// outermost). Empty is a plain row-major buffer.
///
/// A `Tiled` stage groups the final tile, the block a cmma transaction reads unstrided. A final
/// space has no grid left to tile, so it stays plain whatever the layout asks for.
fn stage_nesting(space: &Space, stage: StageStorage) -> Vec<Space> {
    match stage {
        StageStorage::Tiled if !space.is_final() => vec![space.final_space()],
        _ => Vec::new(),
    }
}

/// A dense stage's physical line extents: `[extents…]` flat, or `[grid…, …, block…]`, one grid per
/// level of `nesting`. A level contributes how many of the next block down it holds; the innermost
/// contributes its own extents.
fn storage_extents(space: &Space, vector_size: usize, nesting: &[Space]) -> Vec<usize> {
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
    // Rounded up, not truncated: a padded stage's innermost extent need not fill whole lines, and
    // the spare lanes of the last one are its padding. `fill_extent` refuses the case where the
    // rounding would instead mean the stage and its source disagree; every fill path asks it.
    let last = extents.len() - 1;
    extents[last] = extents[last].div_ceil(vector_size);
    extents
}

/// What a padded fill needs beyond the two boxes: `width` scalar source cells assembled per
/// destination line, and `lanes` the innermost extent past which those cells are padding. `None`
/// lanes is a `Dynamic` extent, where nothing is known at comptime and the source's own bounds
/// check is what zeroes them ([`fill_extent`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    pub(crate) width: usize,
    pub(crate) lanes: Option<usize>,
    /// The physical rank both boxes share, which only this path needs: the 1:1 copy reads its
    /// line whole and never rebuilds a coordinate.
    pub(crate) rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// `16 -> 8 -> 4` on both axes, so the space, its `divide()`, and its `final_space()` are
    /// three distinct block shapes to nest.
    fn space() -> Space {
        Tiling::over(&mut (), &[(M, 16), (N, 16)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.walk(&[(M, 8), (N, 8)]);
            })
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.walk(&[(M, 4), (N, 4)]);
            })
            .build()
    }

    /// [`space`] plus the ungathered innermost axis a gathered projection is required to carry.
    fn gathered_space() -> Space {
        Tiling::over(&mut (), &[(M, 16), (N, 16), (K, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.walk(&[(M, 8), (N, 8), (K, 4)]);
            })
            .build()
    }

    /// No nesting is the plain row-major buffer: the space's own extents, innermost in lines.
    #[test]
    fn flat_nesting_is_the_space_itself() {
        assert_eq!(storage_extents(&space(), 1, &[]), vec![16, 16]);
        assert_eq!(storage_extents(&space(), 4, &[]), vec![16, 4]);
    }

    /// One block is the `[grid…, tile…]` split: each axis holds `16 / 4` tiles of `4`.
    #[test]
    fn one_block_splits_grid_and_tile() {
        let space = space();
        assert_eq!(
            storage_extents(&space, 1, &[space.final_space()]),
            vec![4, 4, 4, 4]
        );
    }

    /// Two nested blocks add a middle grid, each level counting how many of the block below it
    /// it holds: `16 = 2 x (8 = 2 x (4))`.
    #[test]
    fn two_blocks_nest() {
        let space = space();
        let nesting = [space.divide(), space.final_space()];
        let extents = storage_extents(&space, 1, &nesting);
        assert_eq!(extents, vec![2, 2, 2, 2, 4, 4]);
        // The nesting only regroups the buffer, never resizes it.
        assert_eq!(extents.iter().product::<usize>(), space.tile_size());
    }

    /// The strides a buffer's extents imply, row-major: `[grid…, tile…]` or plain, the same rule.
    #[test]
    fn a_form_strides_row_major() {
        let space = space();
        let form = StageForm::dense(&space, 4, StageStorage::Strided);
        assert_eq!(form.extents, vec![16, 4]);
        assert_eq!(form.strides(), vec![4, 1]);
        assert_eq!(form.cells(), space.tile_size() / 4);

        let tiled = StageForm::dense(&space, 1, StageStorage::Tiled);
        assert_eq!(tiled.extents, vec![4, 4, 4, 4]);
        assert_eq!(tiled.strides(), vec![64, 16, 4, 1]);
    }

    /// A gathered stage is the compacted window, not the logical tile: `M` and `N` here map onto
    /// one physical axis, so the stage holds their receptive field instead of their product. `K`
    /// rides identity innermost, which every gathered projection is required to carry
    /// ([`Projection::validate`]).
    #[test]
    fn a_gathered_form_is_the_compacted_window() {
        let space = gathered_space().divide();
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
        let space = gathered_space().divide();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        StageForm::gathered(&space, 1, StageStorage::Tiled, &projection);
    }

    /// A block that does not divide the one enclosing it has no `[grid…, block…]` split.
    #[test]
    #[should_panic(expected = "must divide")]
    fn a_block_must_divide_its_enclosing_block() {
        let space = space();
        // Reversed: the coarse block sits inside the fine one.
        storage_extents(&space, 1, &[space.final_space(), space.divide()]);
    }

    /// A `Tiled` stage groups the final tile; a final space has no grid left, so it stays plain.
    #[test]
    fn stage_nesting_follows_the_layout() {
        let space = space();
        assert_eq!(
            stage_nesting(&space, StageStorage::Tiled),
            vec![space.final_space()]
        );
        assert!(stage_nesting(&space, StageStorage::Strided).is_empty());
        assert!(stage_nesting(&space.final_space(), StageStorage::Tiled).is_empty());
    }
}
