//! Deriving a shared-memory stage from an operand: the [`StageForm`] it takes (physical extents
//! plus the two mappings that address them) and the `smem*` constructors that allocate one.

use cubecl::zspace::SmallVec;
use cubecl::{prelude::*, quant::scheme::QuantScheme, std::quant::view::KnownScale};

use crate::*;

/// The byte alignment a TMA-filled stage's shared buffer must have.
pub(crate) const TMA_STAGE_ALIGNMENT: usize = 128;

#[cube]
impl<T: Numeric> Memory<T> {
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
        Memory::smem_aligned(space, vector_size, storage, units, comptime!(0usize))
    }

    /// [`smem`](Memory::smem) with a minimum byte alignment on the shared
    /// buffer. A TMA-filled stage needs one ([`TMA_STAGE_ALIGNMENT`]); `0`
    /// leaves the buffer at its element alignment.
    pub fn smem_aligned(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] alignment: usize,
    ) -> Tile<T> {
        let elem_bytes = T::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            vector_size,
            storage,
            LineBytes(vector_size * elem_bytes)
        ));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        Memory::smem_with_form(
            space,
            vector_size,
            units,
            form,
            map,
            ComptimeOption::new_None(),
            alignment,
        )
    }

    /// [`smem`](Memory::smem) for a *gathered* operand: the stage holds the physical window its
    /// sub-tile reads, compacted ([`Compaction`]), rather than the logical tile, which would
    /// replicate each physical cell by roughly the tap count; the window holds each one once.
    ///
    /// The stage therefore keeps the operand's own [`Projection`] (with the compaction's lattice
    /// quotiented out) instead of becoming direct, so [`Tile::nd`] and [`at`](Memory::at) address
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
        let source = Memory::<T>::pending_source_window(
            comptime!(form.steps.clone()),
            comptime!(signed),
            comptime!(boundaries),
        );
        // A gathered stage is never a TMA destination (TMA operands are
        // direct), so it takes no extra alignment.
        Memory::smem_with_form(
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
        Memory::smem_over(
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

    /// [`smem`](Memory::smem) over the words a [`packed`](Packing::Packed) operand is stored in:
    /// the line narrows by the packing's factor and the buffer keeps `packing`, so every read
    /// unpacks as one of the global window does. No scales: those are an operand of their own.
    pub(crate) fn smem_packed(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] packing: Packing,
    ) -> Tile<T> {
        let word_bytes = u32::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            vector_size,
            storage,
            LineBytes(packing.physical(vector_size) * word_bytes)
        ));
        let size!(WP) = comptime!(packing.physical(vector_size));
        let smem = Shared::<[Vector<u32, WP>]>::new_slice(comptime!(form.cells()));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        Memory::smem_over(
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

    /// [`smem`](Memory::smem) staging the element `I` an operand is *stored* in (`i8`, or packed
    /// `u32`) rather than the one it serves: the line narrows to `vector_size / pack`, so the stage
    /// is that much smaller and the leaf dequantizes at read, not the fill inflating to `T`.
    ///
    /// Carries a compact `Shared` scales buffer beside the values: one f32 per block of the
    /// sub-tile, refilled per region by [`fill_from`](Memory::fill_from).
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
        let stored_bytes = I::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            vector_size,
            storage,
            LineBytes(vector_size / scheme.num_quants() * stored_bytes)
        ));
        let size!(WP) = comptime!(vector_size / scheme.num_quants());
        let smem = Shared::<[Vector<I, WP>]>::new_slice(comptime!(form.cells()));
        let quant = smem_quant_info(comptime!(space.clone()), table, comptime!(scheme));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        Memory::smem_over(
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
    /// recover it via [`lines_storage`](Memory::lines_storage)) and windows the whole buffer.
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
        // A stage is an address: it is read back by the instruction that consumes it, which is
        // the one thing a sink cannot do.
        let backing = Backing::<T>::new_Buffer(unsafe {
            smem.inner_ref()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        });
        Memory::smem_backed(
            space,
            vector_size,
            units,
            backing,
            quant,
            packing,
            form,
            map,
            source,
            comptime!(Write::Replace),
        )
    }

    /// The whole-buffer window over a shared-memory `backing` laid out as `form`: what every smem
    /// constructor ends in. `write` is what a store into it does, which is a replace for every
    /// stage and an add for [`smem_accumulation`](Tile::smem_accumulation)'s sink.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn smem_backed(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] units: usize,
        backing: Backing<T>,
        quant: ComptimeOption<QuantInfo>,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
        #[comptime] write: Write,
    ) -> Tile<T> {
        let (physical_shape, physical_strides) = storage_layout(comptime!(form.clone()));
        let (origin, extent) = full_window(comptime!(form.clone()));
        // Smem never overhangs its own buffer, so the bound is the extent and checks are off.
        let bound = extent.clone();
        let gmem_projection = comptime!(form.positional.clone());
        Tile::<T> {
            kind: TileKind::new_Memory(Memory::<T> {
                address: comptime!(AddressSpace::Shared),
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
                    rows: comptime!(form.rows),
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
                    write,
                    units,
                    // A stage is allocated here, whole: one storage tile over the buffer.
                    storage: Storage::Strided,
                }),
                unit_share: comptime!(UnitShare::Repeated),
                split_share: comptime!(SplitShare::Whole),
                init_from: comptime!(InitFrom::Cell),
                factor: Factor::none(),
                source_window: source,
                lands: false,
            }),
            place: comptime!(Placement::new(space, 0usize, Vec::new())),
        }
    }

    /// One plane's landing: a dense, scalar stage over `space`, one window per plane of the
    /// cube in one shared buffer, this plane's found by the walk's own decode of the hardware
    /// position. The buffer comes back beside the tile, for the units that fill it.
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
        let start = PLANE_POS.cast::<usize>() * cells;
        let end = start + cells;
        let window =
            Shared::<[T]>::new_slice(comptime!(cells * planes)).map(|all| &all[start..end]);
        let elem_bytes = T::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            1,
            StageStorage::Strided,
            LineBytes(elem_bytes)
        ));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let tile = Memory::smem_over(
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
    /// while the origin and bound are written by each [`fill_from`](Memory::fill_from) from the
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
/// by [`fill_from`](Memory::fill_from), read by [`transparent`](Memory::transparent) as gmem's.
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
        // ([`Memory::stage_scales`]), so the stage serves effective scales under the one-level
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
/// a `Tile::copy_from` caller can get wrong; sizes are [`fill_straight`](Memory::fill_straight)'s.
pub(crate) fn stage_compaction(
    src: &Projection,
    dst: &Projection,
    vector_size: usize,
    space: &Space,
) -> Option<Compaction> {
    // A `Memory` carries the *coordinate*-space map ([`Projection::untiled`]), where storage
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
/// place a dense stage and a gathered one differ, so [`smem_over`](Memory::smem_over) builds
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
    /// Where each line of a block row is kept.
    pub(crate) rows: RowPlacement,
}

impl StageForm {
    /// A materialized dense copy of the logical tile: what every direct operand stages into. An
    /// empty `nesting` is a plain row-major buffer; each block in it adds a `[grid…, block…]`
    /// split, so the buffer lays the innermost block down contiguously.
    ///
    /// `line` is a physical line's size, which is what a block's rows are placed by
    /// ([`RowPlacement`]).
    pub(crate) fn dense(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        line: LineBytes,
    ) -> StageForm {
        let nesting = stage.nesting(space);
        let extents = StageForm::dense_extents(space, vector_size, &nesting);
        let rows = match stage {
            StageStorage::Tiled { chunks, .. } => RowPlacement::new(chunks, &extents, line),
            StageStorage::Strided | StageStorage::Lines { .. } => RowPlacement::InOrder,
        };
        StageForm {
            extents,
            rows,
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
        // `Tiled` comes from a cmma leaf, which `Slot::new` refuses a gathered operand for. The
        // nesting has nowhere to go here, so it is refused rather than silently dropped.
        assert!(
            matches!(stage, StageStorage::Strided),
            "StageForm: a gathered operand stages into a plain row-major window, but {stage:?} \
             storage was asked for"
        );
        let compaction = Compaction::new(projection, vector_size, |axis| space.extent(axis));
        let extents = compaction.line_extents(vector_size);
        StageForm {
            rows: RowPlacement::InOrder,
            positional: Projection::of_tiling(StorageTiling::uniform(extents.len(), 0)),
            projection: compaction.projection().clone(),
            steps: compaction.steps().iter().copied().collect(),
            extents,
        }
    }

    /// How many lines the buffer holds.
    /// The buffer's rank: how many physical axes its layout addresses.
    pub(crate) fn physical_rank(&self) -> usize {
        self.projection.physical_rank()
    }

    pub(crate) fn cells(&self) -> usize {
        self.pitched().iter().product()
    }

    /// Row-major suffix-product strides over [`extents`](StageForm::extents), a row pitched past
    /// its padding.
    fn strides(&self) -> Vec<usize> {
        let pitched = self.pitched();
        (0..pitched.len())
            .map(|p| pitched[p + 1..].iter().product())
            .collect()
    }

    /// [`extents`](StageForm::extents) with each row widened by the lines its placement pads it
    /// with: what the buffer lays down, where the extents are what it holds.
    fn pitched(&self) -> Vec<usize> {
        let mut pitched = self.extents.clone();
        if let Some(last) = pitched.last_mut() {
            *last += self.rows.padding();
        }
        pitched
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
                    "Memory::smem: a {b}-element storage block must divide the {e}-element block \
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
        // and the spare units of the last one are its padding. `fill_extent` refuses the case where
        // the rounding would mean the stage and its source disagree; every fill path asks it.
        let last = extents.len() - 1;
        extents[last] = extents[last].div_ceil(vector_size);
        extents
    }
}

/// A stage's physical shape and strides, in lines like a launched operand's ([`GlobalOperand`]).
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
/// destination line, and `units` the innermost extent past which those cells are padding; `None`
/// for a `Dynamic` extent, where the source's own bounds check zeroes them ([`fill_extent`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    pub(crate) width: usize,
    pub(crate) extent: Option<usize>,
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
            StageStorage::Lines { .. } => {
                panic!("StageStorage::Lines: the plane's units are not shared memory")
            }
            StageStorage::Tiled { block, .. } => {
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
        let form = StageForm::dense(&space, 4, StageStorage::Strided, LineBytes(16));
        assert_eq!(form.extents, vec![16, 4]);
        assert_eq!(form.strides(), vec![4, 1]);
        assert_eq!(form.cells(), space.cells() / 4);

        let tiled = StageForm::dense(
            &space,
            1,
            StageStorage::Tiled {
                block: space.leaf(&levels).extents(),
                chunks: RowChunks::InOrder,
            },
            LineBytes(4),
        );
        assert_eq!(tiled.extents, vec![4, 4, 4, 4]);
        assert_eq!(tiled.strides(), vec![64, 16, 4, 1]);
    }

    /// A swizzled block keeps the extents and strides of one in order, and resolves the swizzle
    /// off its innermost block: rows of four 4-byte lines are one chunk each, left in order; rows
    /// of four 16-byte lines are permuted, their rows and lines on the last two physical axes.
    #[test]
    fn a_swizzled_block_is_laid_out_as_one_in_order() {
        let (space, levels) = space();
        let swizzled = |line_bytes| {
            let line = LineBytes(line_bytes);
            StageForm::dense(
                &space,
                1,
                StageStorage::Tiled {
                    block: space.leaf(&levels).extents(),
                    chunks: RowChunks::Swizzled,
                },
                line,
            )
        };
        assert_eq!(swizzled(4).extents, vec![4, 4, 4, 4]);
        assert_eq!(swizzled(4).strides(), vec![64, 16, 4, 1]);
        assert_eq!(swizzled(4).rows, RowPlacement::InOrder);
        let RowPlacement::Swizzled(swizzle) = swizzled(16).rows else {
            panic!("rows of four 16-byte lines are swizzled")
        };
        assert_eq!((swizzle.row_axis(), swizzle.line_axis()), (2, 3));
    }

    /// A padded block keeps its extents, and its rows are pitched one chunk further apart: the
    /// strides and the size grow, the window does not.
    #[test]
    fn a_padded_block_pitches_its_rows_past_a_chunk() {
        let (space, levels) = space();
        let padded = StageForm::dense(
            &space,
            1,
            StageStorage::Tiled {
                block: space.leaf(&levels).extents(),
                chunks: RowChunks::Padded,
            },
            LineBytes(16),
        );
        assert_eq!(padded.extents, vec![4, 4, 4, 4]);
        assert_eq!(padded.rows, RowPlacement::Padded { lines: 1 });
        assert_eq!(padded.strides(), vec![80, 20, 5, 1]);
        assert_eq!(padded.cells(), 320);
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
                chunks: RowChunks::InOrder,
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
            chunks: RowChunks::InOrder,
        };
        assert!(tiled.nesting(&space)[0] == space.leaf(&levels));
        assert!(StageStorage::Strided.nesting(&space).is_empty());
        assert!(tiled.nesting(&space.leaf(&levels)).is_empty());
    }
}
