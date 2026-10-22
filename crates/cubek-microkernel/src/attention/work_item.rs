use crate::attention::{AttentionOperand, AttentionOperands, D, DV, Q, S};
use crate::matmul::{
    MatmulAxes, MatmulContraction, MatmulDestination, MatmulLayout, MatmulOperands, MatmulOrder,
    MatmulRows, MatmulWorkItem,
};
use crate::staging::StagedOperand;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::{Axis, Space};

/// A work item of this kind as its microkernels see it: the elements it reads and writes, the
/// one it sums in, the query heads that share each key head, and the dims a row of the query and
/// of the output spans, which every owner holds whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionWorkItem {
    pub elements: AttentionOperands<ElemType>,
    pub sum: ElemType,
    /// Which elements the query, keys and values may be read in.
    pub reading: AttentionReading,
    /// Positions a page of the cache holds; `None` where the cache is not paged.
    pub page_size: Option<usize>,
    /// Query heads per key head.
    pub group: usize,
    pub head: usize,
    pub value: usize,
}

/// Which elements an attention's operands may be read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionReading {
    /// As stored, or rounded to an element a matrix instruction takes.
    Rounded,
    /// As stored only: a pool read through a page table is bound as stored, never as words.
    AsStored,
}

impl AttentionWorkItem {
    /// The elements a kernel of this kind defines, in the order it names them: the sum, then the
    /// query's, the one its operands are read at.
    pub fn kernel_elements(&self) -> [ElemType; 2] {
        [self.sum, self.elements.query]
    }

    /// The elements a kernel that may read its operands at another element than they are stored
    /// in defines, in the order it names them: the sum, the output's, and the one the query, the
    /// keys and the values are read in. Where the last two differ the operands are bound as their
    /// `u32` words, a value each, which the kernel's stages decode at the read element; their
    /// arguments say so.
    pub fn served_kernel_elements(&self) -> [ElemType; 3] {
        [self.sum, self.elements.out, self.elements.query]
    }

    /// The elements its query, keys and values may reach a matrix instruction in, the exact one
    /// first: as stored, and `f32` also rounded to tf32, ten bits of mantissa, the instruction an
    /// NVIDIA GPU offers `f32` operands, the sum staying `f32`, where the reading allows it.
    pub fn readings(&self) -> impl Iterator<Item = ElemType> {
        let stored = self.elements.query;
        let rounded = match stored == ElemType::Float(FloatKind::F32)
            && self.reading == AttentionReading::Rounded
        {
            true => Some(ElemType::Float(FloatKind::TF32)),
            false => None,
        };
        core::iter::once(stored).chain(rounded)
    }

    /// This work item with its query, keys and values read in `read`; the output stays as it is
    /// stored.
    pub fn read_as(&self, read: ElemType) -> AttentionWorkItem {
        AttentionWorkItem {
            elements: AttentionOperands {
                query: read,
                key: read,
                value: read,
                out: self.elements.out,
            },
            ..*self
        }
    }

    /// The score, `query · keyᵀ`, as a shared multiplication microkernel is handed it: rows `Q` by
    /// columns `S`, contracting `D`, into a score of `score` elements its owner keeps. The keys lie
    /// contiguous along `D`, the contraction, as a col weight does.
    pub fn score(&self, score: ElemType) -> MatmulWorkItem {
        MatmulWorkItem {
            contraction: MatmulContraction::AsStored,
            elements: MatmulOperands {
                lhs: self.elements.query,
                rhs: self.elements.key,
                out: score,
            },
            sum: self.sum,
            layouts: MatmulOperands {
                rhs: MatmulLayout::Col,
                ..MatmulOperands::all(MatmulLayout::Row)
            },
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes { m: Q, n: S, k: D },
        }
    }

    /// The value, `probabilities · value`, as a shared multiplication microkernel is handed it:
    /// rows `Q` by columns `DV`, contracting `S`, out of probabilities of `probabilities` elements
    /// its owner keeps. The values lie contiguous along `DV`, the columns, as a row weight does.
    pub fn value(&self, probabilities: ElemType) -> MatmulWorkItem {
        MatmulWorkItem {
            contraction: MatmulContraction::AsStored,
            elements: MatmulOperands {
                lhs: probabilities,
                rhs: self.elements.value,
                out: self.elements.out,
            },
            sum: self.sum,
            layouts: MatmulOperands {
                rhs: MatmulLayout::Row,
                ..MatmulOperands::all(MatmulLayout::Row)
            },
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes { m: Q, n: DV, k: S },
        }
    }

    /// One stage of `keys` keys and their values, each over its own axes and bound innermost
    /// along the axis its lines run along, the keys in `key_order`.
    pub fn stage(&self, keys: usize, key_order: MatmulOrder) -> [StagedOperand; 2] {
        let operand = |operand: AttentionOperand, row: Axis, extent: usize| StagedOperand {
            element: self.elements.get(operand),
            space: Space::new(&[(S, keys), (row, extent)]),
            innermost: operand.innermost(key_order),
        };
        [
            operand(AttentionOperand::Key, D, self.head),
            operand(AttentionOperand::Value, DV, self.value),
        ]
    }
}
