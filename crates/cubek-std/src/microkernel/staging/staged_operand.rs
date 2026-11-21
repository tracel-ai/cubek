use crate::Hardware;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::{Dimension, Space, StageStorage, stage::Prefetch};

/// One operand's box of a stage, as the ring that holds it reads it: what a stage of it costs in
/// shared memory, and in registers where the next one waits there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedOperand {
    /// The element the stage holds it in.
    pub element: ElemType,
    /// Its box of one stage, over its own dimensions.
    pub space: Space,
    /// The dimension it is bound innermost along, which the stage's rows run along.
    pub innermost: Dimension,
    /// What its fill does to it on the way into the stage.
    pub fill: StageFill,
}

/// What a fill does to an operand on its way into the stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageFill {
    /// Copies it as it is bound, at its own width.
    AsStored,
    /// Converts it: rounds tf32 or decodes e2m1 stored as its own element. The register prefetch
    /// does not carry that.
    Converted,
    /// Decodes a packed input under its scales, which the register prefetch carries: it holds the
    /// packed loads and their scales across a contraction and decodes them into the stage after.
    Decoded,
}

impl StagedOperand {
    /// An operand stored in `element` whose fill converts it only where no buffer stores
    /// `element`, e2m1 or tf32.
    pub fn new(element: ElemType, space: Space, innermost: Dimension) -> Self {
        let in_words = [FloatKind::E2M1, FloatKind::TF32]
            .map(ElemType::Float)
            .contains(&element);
        Self {
            element,
            space,
            innermost,
            fill: match in_words {
                true => StageFill::Converted,
                false => StageFill::AsStored,
            },
        }
    }

    /// This operand with a fill that decodes it, as a fill decoding a packed input does.
    pub fn decoded_by_fill(self) -> Self {
        Self {
            fill: StageFill::Decoded,
            ..self
        }
    }

    /// This operand with a fill that converts it, as one rounding it to tf32 or reading it at
    /// another element does.
    pub fn converted_by_fill(self) -> Self {
        Self {
            fill: StageFill::Converted,
            ..self
        }
    }

    /// Elements one stage holds of it.
    pub fn cells(&self) -> usize {
        self.space
            .dimensions()
            .map(|dimension| self.space.extent(dimension))
            .product()
    }

    /// Bytes one stage holds of it on `hw`, laid out as `storage`, at its most over every width it
    /// could bind at: the powers of two up to the widest line `hw` reads that its rows hold whole
    /// lines of, each line whole bytes. Asked before the operands' widths are settled.
    pub fn bytes(&self, hw: &Hardware, storage: &StageStorage) -> usize {
        let rows = self.rows();
        let row = self.space.extent(self.innermost);
        let bits = self.element.size_bits();
        self.widths(hw)
            .filter(|&width| row.is_multiple_of(width) && (width * bits).is_multiple_of(8))
            .map(|width| storage.bytes(&rows, width, bits))
            .max()
            .expect("a row holds whole bytes")
    }

    /// Its box with the dimension it is bound innermost along last, which the stage's rows run
    /// along.
    fn rows(&self) -> Space {
        let dimensions: Vec<(Dimension, usize)> = self
            .space
            .dimensions()
            .filter(|&dimension| dimension != self.innermost)
            .chain([self.innermost])
            .map(|dimension| (dimension, self.space.extent(dimension)))
            .collect();
        Space::new(&dimensions)
    }

    /// Scalars one of `units` units holds of a stage of it fetched into registers, at its most
    /// over every width it could bind at on `hw`: the powers of two up to the widest line `hw`
    /// reads. Asked before the operands' widths are settled.
    pub fn most_prefetched(&self, hw: &Hardware, units: usize) -> usize {
        self.widths(hw)
            .map(|width| Prefetch::scalars(self.cells(), units.max(1), width))
            .max()
            .expect("the widths hold one element")
    }

    /// Every width it could bind at on `hw`: the powers of two up to the widest line `hw` reads.
    fn widths(&self, hw: &Hardware) -> impl Iterator<Item = usize> {
        (0..=hw.vector_ceiling(self.element).ilog2()).map(|power| 1 << power)
    }
}
