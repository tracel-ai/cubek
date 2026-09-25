//! Deriving the tile an operand is staged into: which element the stage takes, which kind it is
//! (shared memory, or the plane's own units), and the cooperative evaluation that fills a stage
//! from a procedural source.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// Cooperatively materialize a coordinate-backed source into this plain, direct scalar memory
    /// tile. Workers write cyclic positions across it, so the caller must ensure every unit in the
    /// cube owns this window: a property of the level's distribution, not of buffer coverage.
    pub(crate) fn fill_procedural(&mut self, src: &Procedural<T>, #[comptime] space: Space) {
        comptime!(assert!(
            self.store.packing == Packing::Plain
                && self.projection.is_direct()
                && self.store.vector_size == 1,
            "Memory::fill_procedural: procedural sources require a plain, direct scalar destination"
        ));
        // Read the destination's runtime window rather than the comptime space so direct copies
        // also work when another operand witnesses a Dynamic extent.
        let shape = self.window.extent.clone();
        let mut dst = self.flat_mut::<Const<1>>();
        let total = dst.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let pos = shape.unravel(i.cast::<u32>());
            // TODO: staging cannot see its consumer, so this masked fill uses Procedural's
            // zero fallback, not every reduction's identity (Max on negatives, Min on positives).
            // A reduction-aware contract must carry validity or the consumer's identity here.
            dst.write(
                i,
                Vector::cast_from(src.read_masked(&pos, comptime!(space.clone()))),
            );
            i += workers;
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of `operand`,
    /// laid out as `storage` and, where `width` is stated, served in lines that wide (an axis gmem
    /// could not vectorize still reaches the leaf in lines). Both are the stages' to state.
    ///
    /// The stage takes the element the operand needs staged: the one it *serves* when the load
    /// decodes it ([`DequantAt::Load`], always so for a plain operand), else the one it is *stored*
    /// in ([`smem_stored`](Memory::smem_stored)). The operand carries which, so no caller asks.
    pub(crate) fn stage(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
    ) -> Tile<T> {
        match comptime!(storage.clone()) {
            // The units are not memory: the plane holds them, at the depth one region of
            // `level` sits, so the regions below the level window them as they window the
            // operand.
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
        let dequant_at = operand.dequant_at();
        match comptime!(dequant_at) {
            DequantAt::Load => {
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
                // A TMA-filled stage's shared buffer must be TMA-aligned (its bulk copy addresses
                // shared memory directly); a copy-filled one needs only the element alignment
                // `smem` gives. TMA operands are always direct, so the gathered arm never needs it.
                let delivery = operand.delivery();
                let alignment = comptime!(match delivery.is_tma() {
                    true => TMA_STAGE_ALIGNMENT,
                    false => 0usize,
                });
                if comptime!(projection.is_direct()) {
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
            DequantAt::Read => Memory::smem_stored(operand, level, storage),
        }
    }

    /// [`stage`](Memory::stage) in the element the operand is *stored* in rather than the one it
    /// serves, under [`DequantAt::Read`]: a quantized operand keeps its stored form (`i8` or packed
    /// `u32`) plus scales ([`smem_quant`](Memory::smem_quant)), and the leaf dequantizes on read.
    fn smem_stored(
        operand: &Tile<T>,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
    ) -> Tile<T> {
        let space = comptime!(level.child(&operand.place.space));
        let vector_size = operand.vector_size();
        let units = operand.units();
        match &operand.kind {
            TileKind::Memory(g) => {
                #[comptime]
                match &g.store.quant {
                    // No scheme: the words as they lie where the operand is packed, which is
                    // the only stored form a scheme-less operand has, else a plain stage.
                    ComptimeOption::None => match comptime!(g.store.packing) {
                        Packing::Plain => Memory::smem(space, vector_size, storage, units),
                        packing => Memory::smem_packed(space, vector_size, storage, units, packing),
                    },
                    // The store was allocated at the scheme's own storage ([`scheme_packing`]).
                    ComptimeOption::Some(info) => match comptime!(g.store.packing) {
                        Packing::Native => Memory::smem_quant::<i8>(
                            space,
                            vector_size,
                            storage,
                            units,
                            info.table.clone(),
                            comptime!(info.scheme),
                        ),
                        Packing::Packed { field: _ } => Memory::smem_quant::<u32>(
                            space,
                            vector_size,
                            storage,
                            units,
                            info.table.clone(),
                            comptime!(info.scheme),
                        ),
                        Packing::Plain => {
                            panic!("Memory::smem_stored: a quantized store is never plain")
                        }
                    },
                }
            }
            // A tma source has no stored form to keep: it carries no scheme (`quantized` is a
            // strided-builder knob, and a tma tile is scalar), so served == stored. Giving it
            // one must not reuse this arm; see `Slot::new`, which refuses that combination.
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
