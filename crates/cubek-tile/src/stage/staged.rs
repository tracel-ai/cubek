//! Allocating the tile an operand is staged into, and filling a stage from a procedural source.

use cubecl::prelude::*;

use crate::*;

/// Which element a stage holds: the values the operand serves, or its stored form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StageElement {
    /// Served values, decoded as they land.
    Served,
    /// The stored form (words for a packed operand).
    Stored,
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// Cooperatively materialize a procedural source into this plain, direct scalar tile.
    /// Every unit in the cube must own this window.
    pub(crate) fn fill_procedural(&mut self, src: &Procedural<T>, #[comptime] space: Space) {
        comptime!(assert!(
            self.store.packing == Packing::Plain
                && self.projection.is_direct()
                && self.store.vector_size == 1,
            "Memory::fill_procedural: procedural sources require a plain, direct scalar destination"
        ));
        // Runtime window, so a Dynamic extent witnessed by another operand still works.
        let shape = self.window.extent.clone();
        let mut dst = self.flat_mut::<Const<1>>();
        let total = dst.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let pos = shape.unravel(i.cast::<u32>());
            // TODO: masked cells use Procedural's zero fallback, not the reduction's identity.
            dst.write(
                i,
                Vector::cast_from(src.read_masked(&pos, comptime!(space.clone()))),
            );
            i += workers;
        }
    }

    /// Allocate a tile staging one `divide()` sub-tile of `operand`, laid out as `storage` and
    /// served `width` wide where stated.
    pub(crate) fn stage(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
    ) -> Tile<T> {
        match comptime!(storage.clone()) {
            // The plane holds the units at the depth of one region of `level`.
            StageStorage::Lines { read } => Tile::<T> {
                kind: TileKind::new_Lines(Lines::<T>::new(
                    operand,
                    comptime!(level.clone()),
                    comptime!(read),
                )),
                place: comptime!(Placement::new(
                    level.child(&operand.place.space),
                    operand.place.depth + 1,
                    operand.place.levels.clone()
                )),
            },
            _ => Memory::<T>::stage_memory(operand, level, storage, width),
        }
    }

    /// [`stage`](Memory::stage) in shared memory.
    fn stage_memory(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
    ) -> Tile<T> {
        // Scaled or looked-up sources are decoded by the fill, so the stage holds served values.
        let decodes = operand.scaled();
        let stored = operand.stage_element();
        match comptime!(if decodes {
            StageElement::Served
        } else {
            stored
        }) {
            StageElement::Served => {
                let space = comptime!(level.child(&operand.place.space));
                let projection = operand.projection();
                let units = operand.units();
                let source_width = operand.vector_size();
                let vector_size = comptime!(match width {
                    Some(width) => {
                        assert!(
                            source_width == 1,
                            "Memory::stage: a padded stage assembles its lines from scalar \
                             source cells, so the operand it pads must be unvectorized (it is \
                             served {source_width} wide)"
                        );
                        assert!(
                            width > 1,
                            "Memory::stage: a padded stage width must widen the operand's own \
                             1-wide lines (got {width})"
                        );
                        width
                    }
                    None => source_width,
                });
                // A TMA-filled stage's buffer must be TMA-aligned; TMA operands are always direct.
                let delivery = operand.delivery();
                let alignment = comptime!(match delivery.is_tma() {
                    true => TMA_STAGE_ALIGNMENT,
                    false => 0usize,
                });
                // A decoded stage is dense over the values' axes whatever the source's map.
                if comptime!(projection.is_direct() || decodes) {
                    Memory::smem_aligned(space, vector_size, storage, units, alignment)
                } else {
                    Memory::smem_gathered(
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
            StageElement::Stored => Memory::smem_stored(operand, level, storage),
        }
    }

    /// [`stage`](Memory::stage) in the element the operand is stored in.
    fn smem_stored(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
    ) -> Tile<T> {
        let space = comptime!(level.child(&operand.place.space));
        let vector_size = operand.vector_size();
        let units = operand.units();
        match &operand.kind {
            TileKind::Memory(g) => match comptime!(g.store.packing) {
                Packing::Plain => Memory::smem(space, vector_size, storage, units),
                packing => Memory::smem_packed(space, vector_size, storage, units, packing),
            },
            TileKind::TmaGmem(_) => Memory::smem(space, vector_size, storage, units),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Memory::smem_stored: a fragment is not a stage source")
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!(
                    "Memory::smem_stored: a procedural tile and the plane's units are not a stage source"
                )
            }
        }
    }
}
