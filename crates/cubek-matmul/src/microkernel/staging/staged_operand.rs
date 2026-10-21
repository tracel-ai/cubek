use cubecl::ir::ElemType;
use cubek_std::Hardware;
use cubek_tile::{
    Axis, Space, StageStorage,
    stage::{Prefetch, RowChunks},
};

/// One operand's box of a stage, as the ring that holds it reads it: what a stage of it costs in
/// shared memory, and in registers where the next one waits there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedOperand {
    /// The element the stage holds it in.
    pub element: ElemType,
    /// Its box of one stage, over its own axes.
    pub space: Space,
    /// The axis it is bound innermost along, which the stage's rows run along.
    pub innermost: Axis,
}

impl StagedOperand {
    /// Elements one stage holds of it.
    pub fn cells(&self) -> usize {
        self.space
            .axes()
            .map(|axis| self.space.extent(axis))
            .product()
    }

    /// Bytes one stage holds of it on `hw`, laid out as `storage`: a padded block's rows each one
    /// chunk longer, or one of `hw`'s widest lines where that is longer, which bounds what cubek
    /// pads them by whatever width the operand binds at.
    pub fn bytes(&self, hw: &Hardware, storage: &StageStorage) -> usize {
        let cells = self.cells();
        let padding = match storage {
            StageStorage::Tiled {
                block,
                chunks: RowChunks::Padded,
            } => {
                let row = block
                    .iter()
                    .find(|&&(axis, _)| axis == self.innermost)
                    .map_or(self.space.extent(self.innermost), |&(_, extent)| extent);
                let per_row = RowChunks::CHUNK_BYTES
                    .max(hw.vector_ceiling(self.element) * self.element.size());
                cells / row * per_row
            }
            StageStorage::Tiled { .. } | StageStorage::Strided | StageStorage::Lines { .. } => 0,
        };
        // Bits rather than bytes: an `e2m1` value is half of one.
        cells * self.element.size_bits() / 8 + padding
    }

    /// Scalars one of `units` units holds of a stage of it fetched into registers, at its most
    /// over every width it could bind at on `hw`: the powers of two up to the widest line `hw`
    /// reads. Asked before the operands' widths are settled.
    pub fn most_prefetched(&self, hw: &Hardware, units: usize) -> usize {
        let widest = hw.vector_ceiling(self.element);
        (0..=widest.ilog2())
            .map(|power| Prefetch::scalars(self.cells(), units.max(1), 1 << power))
            .max()
            .expect("the widths hold one element")
    }
}
