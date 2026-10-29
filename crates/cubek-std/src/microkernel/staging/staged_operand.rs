use crate::Hardware;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::{Axis, Space, StageStorage, stage::Prefetch};

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
    /// What its fill does to it on the way into the stage.
    pub fill: StageFill,
}

/// What a fill does to an operand on its way into the stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageFill {
    /// Copies it as it is bound, at its own width.
    AsStored,
    /// Converts it: rounds tf32, decodes e2m1 or decodes a packed input. The register prefetch
    /// reads a plain source at its own width and does not carry that.
    Converted,
}

impl StagedOperand {
    /// An operand stored in `element` whose fill converts it only where no buffer stores
    /// `element`, e2m1 or tf32.
    pub fn new(element: ElemType, space: Space, innermost: Axis) -> Self {
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

    /// This operand with a fill that converts it, as a fill decoding a packed input does.
    pub fn converted_by_fill(self) -> Self {
        Self {
            fill: StageFill::Converted,
            ..self
        }
    }

    /// Elements one stage holds of it.
    pub fn cells(&self) -> usize {
        self.space
            .axes()
            .map(|axis| self.space.extent(axis))
            .product()
    }

    /// Bytes one stage holds of it on `hw`, laid out as `storage`, at its most over every width it
    /// could bind at: the powers of two up to the widest line `hw` reads that its rows hold whole
    /// lines of, each line whole bytes. Asked before the operands' widths are settled.
    pub fn bytes(&self, hw: &Hardware, storage: &StageStorage) -> usize {
        let rows = self.rows();
        let row = self.space.extent(self.innermost);
        let bits = self.element.size_bits();
        let widest = hw.vector_ceiling(self.element);
        (0..=widest.ilog2())
            .map(|power| 1 << power)
            .filter(|&width: &usize| row.is_multiple_of(width) && (width * bits).is_multiple_of(8))
            .map(|width| storage.bytes(&rows, width, bits))
            .max()
            .expect("a row holds whole bytes")
    }

    /// Its box with the axis it is bound innermost along last, which the stage's rows run along.
    fn rows(&self) -> Space {
        let axes: Vec<(Axis, usize)> = self
            .space
            .axes()
            .filter(|&axis| axis != self.innermost)
            .chain([self.innermost])
            .map(|axis| (axis, self.space.extent(axis)))
            .collect();
        Space::new(&axes)
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
