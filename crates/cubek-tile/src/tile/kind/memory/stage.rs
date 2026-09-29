//! Shared-memory stages: the [`StageForm`] a stage takes and the `smem*` constructors.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::*;

/// The byte alignment a TMA-filled stage's shared buffer must have.
pub(crate) const TMA_STAGE_ALIGNMENT: usize = 128;

#[cube]
impl<T: Numeric> Memory<T> {
    /// Allocate a shared-memory tile over `space` at physical `vector_size`.
    /// `units` is the launch's cube size, `0` when unknown.
    pub(crate) fn smem(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
    ) -> Tile<T> {
        Memory::smem_aligned(space, vector_size, storage, units, comptime!(0usize))
    }

    /// [`smem`](Memory::smem) with a minimum byte alignment on the buffer, never below one
    /// [`RowChunks::CHUNK_BYTES`] chunk: an `ldmatrix` row address needs it.
    pub(crate) fn smem_aligned(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] alignment: usize,
    ) -> Tile<T> {
        let alignment = comptime!(alignment.max(RowChunks::CHUNK_BYTES));
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

    /// [`smem`](Memory::smem) for a gathered operand: the compacted physical window its sub-tile
    /// reads ([`Compaction`]), still addressed by the operand's own [`Projection`].
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

    /// The body every plain smem constructor shares; `alignment` `0` is the element's own.
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
            comptime!(Packing::Plain),
            form,
            map,
            source,
        )
    }

    /// [`smem`](Memory::smem) over the words a [`packed`](Packing::Packed) operand is stored in.
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
            comptime!(packing),
            form,
            map,
            ComptimeOption::new_None(),
        )
    }

    /// The body every smem constructor shares: scalar-erases `smem` and windows the whole buffer.
    #[allow(clippy::too_many_arguments)]
    fn smem_over<S: CubePrimitive>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] units: usize,
        smem: &Shared<[S]>,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
    ) -> Tile<T> {
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
            packing,
            form,
            map,
            source,
            comptime!(Write::Replace),
        )
    }

    /// The whole-buffer window over a shared-memory `backing` laid out as `form`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn smem_backed(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] units: usize,
        backing: Backing<T>,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
        #[comptime] write: Write,
    ) -> Tile<T> {
        let (physical_shape, physical_strides) = storage_layout(comptime!(form.clone()));
        let (origin, extent) = full_window(comptime!(form.clone()));
        // Smem never overhangs its own buffer, so the bound is the extent.
        let bound = extent.clone();
        let gmem_projection = comptime!(form.positional.clone());
        Tile::<T> {
            kind: TileKind::new_Memory(Memory::<T> {
                address: comptime!(AddressSpace::Shared),
                store: Store::<T>::untiled(backing, vector_size, packing),
                layout: BufferLayout {
                    physical_shape,
                    physical_strides,
                    projection: gmem_projection,
                    rows: comptime!(form.rows),
                },
                // Empty: a list over the buffer's own dims would misalign one level down.
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
                    storage: Storage::Strided,
                }),
                unit_share: comptime!(UnitShare::Repeated),
                split_share: comptime!(SplitShare::Whole),
                init_from: comptime!(InitFrom::Cell),
                factor: Factor::none(),
                codebook: Codebook::none(),
                source_window: source,
                lands: false,
            }),
            place: comptime!(Placement::new(space, 0usize, Vec::new())),
        }
    }

    /// One plane's landing: a dense scalar stage over `space`, one window per plane in one buffer.
    /// The buffer is returned beside the tile.
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
            comptime!(Packing::Plain),
            form,
            map,
            ComptimeOption::new_None(),
        );
        (tile, window)
    }

    /// An unfilled [`SourceWindow`] for a gathered stage; each [`fill_from`](Memory::fill_from)
    /// writes its origin and bound.
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

/// The whole-buffer window of a stage: zero origin, its own physical extents.
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

/// The [`Compaction`] of a gathered source's map, which `dst` must be addressed by; `None` when
/// neither side gathers. Panics if `dst` is not that compaction's projection.
pub(crate) fn stage_compaction(
    src: &Projection,
    dst: &Projection,
    vector_size: usize,
    space: &Space,
) -> Option<Compaction> {
    // Direct in coordinate space means no gather, tiled buffers included.
    if src.is_direct() && dst.is_direct() {
        return None;
    }
    // A partition stages like a direct operand: nothing aliases.
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

/// A stage's buffer: its physical extents and the two mappings that address them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct StageForm {
    /// Physical extents in lines.
    extents: Vec<usize>,
    /// The buffer's own per-position map.
    positional: Projection,
    /// How the staged tile's logical axes address those extents.
    projection: Projection,
    /// Per physical axis, what a stage coordinate is multiplied by to land on the source.
    steps: SmallVec<[usize; Space::MAX_RANK]>,
    /// Where each line of a block row is kept.
    pub(crate) rows: RowArrangement,
}

impl StageForm {
    /// A dense copy of the logical tile, one `[grid…, block…]` split per `nesting` block.
    pub(crate) fn dense(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        line: LineBytes,
    ) -> StageForm {
        let nesting = stage.nesting(space);
        let extents = StageForm::dense_extents(space, vector_size, &nesting);
        let rows = match stage {
            StageStorage::Tiled { chunks, .. } => RowArrangement::new(chunks, &extents, line),
            StageStorage::Strided | StageStorage::Lines { .. } => RowArrangement::InOrder,
        };
        StageForm {
            extents,
            rows,
            positional: StageForm::positional(space.rank(), nesting.len() + 1),
            // A dense stage addresses its own buffer directly, whatever its operand was gathered
            // through.
            projection: Projection::direct_over(space),
            steps: SmallVec::new(),
        }
    }

    /// The compacted row-major window a gathered operand stages into ([`Compaction`]).
    fn gathered(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        projection: &Projection,
    ) -> StageForm {
        assert!(
            matches!(stage, StageStorage::Strided),
            "StageForm: a gathered operand stages into a plain row-major window, but {stage:?} \
             storage was asked for"
        );
        let compaction = Compaction::new(projection, vector_size, |axis| space.extent(axis));
        let extents = compaction.line_extents(vector_size);
        StageForm {
            rows: RowArrangement::InOrder,
            positional: StageForm::positional(extents.len(), 1),
            projection: compaction.projection().clone(),
            steps: compaction.steps().iter().copied().collect(),
            extents,
        }
    }

    /// The buffer's rank: how many physical axes its layout addresses.
    pub(crate) fn physical_rank(&self) -> usize {
        self.projection.physical_rank()
    }

    pub(crate) fn cells(&self) -> usize {
        self.pitched().iter().product()
    }

    /// Row-major strides over [`extents`](StageForm::extents), rows pitched past their padding.
    fn strides(&self) -> Vec<usize> {
        let pitched = self.pitched();
        (0..pitched.len())
            .map(|p| pitched[p + 1..].iter().product())
            .collect()
    }

    /// [`extents`](StageForm::extents) with each row widened by its padding lines.
    fn pitched(&self) -> Vec<usize> {
        let mut pitched = self.extents.clone();
        if let Some(last) = pitched.last_mut() {
            *last += self.rows.padding();
        }
        pitched
    }

    /// A stage's buffer addressed by position: `rank` synthetic axes, each split into `pieces` dims.
    fn positional(rank: usize, pieces: usize) -> Projection {
        let axes: Vec<Axis> = (0..rank).map(|p| Axis(p as u8)).collect();
        let labels = StoragePartitioning::level_major(&axes, &vec![pieces; rank]);
        Projection::tiled(&axes, &labels)
    }

    /// A dense stage's physical line extents: `[extents…]` or `[grid…, …, block…]`.
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
        // Rounded up: the last line's spare units are padding.
        let last = extents.len() - 1;
        extents[last] = extents[last].div_ceil(vector_size);
        extents
    }
}

/// A stage's physical shape and strides, in lines.
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

/// What a padded fill needs beyond the two boxes: source cells per line and the padding extent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    pub(crate) width: usize,
    pub(crate) extent: Option<usize>,
    /// The physical rank both boxes share.
    pub(crate) rank: usize,
}

impl StageStorage {
    /// The storage-tiling nesting a stage over `space` gets, coarse to fine; empty is row-major.
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

    /// `16 -> 8 -> 4` on both axes: three distinct block shapes.
    fn space() -> (Space, Vec<Level>) {
        (
            Space::new(&[(M, 16), (N, 16)]),
            Levels::leaf(&[(M, 4), (N, 4)])
                .walk(&[(M, 2), (N, 2)])
                .walk_every(&[M, N])
                .build(),
        )
    }

    /// [`space`] plus an ungathered innermost axis, cut to `8 x 8 x 4`.
    fn gathered_space() -> Space {
        Space::new(&[(M, 8), (N, 8), (K, 4)])
    }

    /// No nesting is the plain row-major buffer.
    #[test]
    fn flat_nesting_is_the_space_itself() {
        let (space, _) = space();
        assert_eq!(StageForm::dense_extents(&space, 1, &[]), vec![16, 16]);
        assert_eq!(StageForm::dense_extents(&space, 4, &[]), vec![16, 4]);
    }

    /// One block is the `[grid…, tile…]` split.
    #[test]
    fn one_block_splits_grid_and_tile() {
        let (space, levels) = space();
        assert_eq!(
            StageForm::dense_extents(&space, 1, &[space.leaf(&levels)]),
            vec![4, 4, 4, 4]
        );
    }

    /// Two nested blocks add a middle grid: `16 = 2 x (8 = 2 x (4))`.
    #[test]
    fn two_blocks_nest() {
        let (space, levels) = space();
        let nesting = [levels[0].child(&space), space.leaf(&levels)];
        let extents = StageForm::dense_extents(&space, 1, &nesting);
        assert_eq!(extents, vec![2, 2, 2, 2, 4, 4]);
        assert_eq!(extents.iter().product::<usize>(), space.cells());
    }

    /// A buffer's extents imply row-major strides.
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

    /// A swizzled block keeps in-order extents and strides, swizzling only 16-byte lines.
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
        assert_eq!(swizzled(4).rows, RowArrangement::InOrder);
        let RowArrangement::Swizzled(swizzle) = swizzled(16).rows else {
            panic!("rows of four 16-byte lines are swizzled")
        };
        assert_eq!((swizzle.row_axis(), swizzle.line_axis()), (2, 3));
    }

    /// A padded block pitches its rows one chunk further apart.
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
        assert_eq!(padded.rows, RowArrangement::Padded { lines: 1 });
        assert_eq!(padded.strides(), vec![80, 20, 5, 1]);
        assert_eq!(padded.cells(), 320);
    }

    /// A gathered stage is the compacted window, not the logical tile.
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
        // 8 x 8 logical cells over 1 + 7 + 7 physical ones, times 4 of `K`.
        assert_eq!(form.extents, vec![15, 4]);
        assert_eq!(form.cells(), 60);
        assert_eq!(form.projection, projection);
        assert!(form.positional.is_direct());
    }

    /// A gathered operand's stage refuses tiled storage.
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

    /// A block must divide the one enclosing it.
    #[test]
    #[should_panic(expected = "must divide")]
    fn a_block_must_divide_its_enclosing_block() {
        let (space, levels) = space();
        // Reversed: the coarse block sits inside the fine one.
        StageForm::dense_extents(&space, 1, &[space.leaf(&levels), levels[0].child(&space)]);
    }

    /// A `Tiled` stage groups the stated block, unless the space already is it.
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
