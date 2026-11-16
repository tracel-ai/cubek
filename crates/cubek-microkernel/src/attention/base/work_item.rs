use crate::attention::{AttentionOperand, AttentionOperands, B, D, DV, G, KV, Q, S};
use crate::matmul::{
    MatmulAxes, MatmulDestination, MatmulLayout, MatmulOperands, MatmulReading, MatmulRows,
    MatmulWorkItem,
};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::{Axis, Space};
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::StagedOperand;

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
    pub head_dim: usize,
    pub val_dim: usize,
}

/// Which elements an attention's operands may be read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionReading {
    /// As stored, or rounded to an element a matrix instruction takes.
    Rounded,
    /// The keys and values as stored only: a pool read through a page table is bound as stored,
    /// never as words. The query, which no table places, may still be converted.
    AsStored,
}

/// The elements an attention's operands reach its two matrix instructions in: the query and the
/// keys reach the score's in `score`; the values, and the probabilities cast to sit beside them,
/// reach the value's in `value`. Each side of a cache may be stored in its own element, as an FP8
/// cache's `e4m3` keys beside 16-bit values, so the two may differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionRead {
    pub score: ElemType,
    pub value: ElemType,
}

impl AttentionRead {
    /// Every operand read in `element`.
    pub fn all(element: ElemType) -> Self {
        Self {
            score: element,
            value: element,
        }
    }
}

impl AttentionWorkItem {
    /// The elements a kernel of this kind defines, in the order it names them: the sum, then the
    /// query's, the one its operands are read at.
    pub fn kernel_elements(&self) -> [ElemType; 2] {
        [self.sum, self.elements.query]
    }

    /// The elements a kernel that reads each operand at its own element defines, in the order it
    /// names them: the sum, the query's, the keys', the values' and the output's. A shuffle widens
    /// each into the sum as it reads it, so an FP8 cache's `e4m3` keys sit beside 16-bit values.
    pub fn operand_elements(&self) -> [ElemType; 5] {
        let AttentionOperands {
            query,
            key,
            value,
            out,
        } = self.elements;
        [self.sum, query, key, value, out]
    }

    /// Whether its query, keys and values are one element: what a microkernel that computes in
    /// its operands' element reads. An FP8 score's `e4m3` query and keys beside 16-bit values
    /// are not: the plane fragments multiply each on its own instruction, and the plane shuffle
    /// widens each into the sum as it reads it.
    pub fn in_one_element(&self) -> bool {
        let AttentionOperands {
            query, key, value, ..
        } = self.elements;
        query == key && key == value
    }

    /// The elements a kernel that writes the output from partial sums defines, in the order it
    /// names them: the sum, then the output's. The output's is the query's but beside an FP8
    /// score, where the output takes the values' element.
    pub fn written_elements(&self) -> [ElemType; 2] {
        [self.sum, self.elements.out]
    }

    /// The elements a kernel that may read its operands at another element than they are stored
    /// in defines, in the order it names them: the sum, the output's, the one the query and the
    /// keys are read in, and the one the values are read in. Where a read element differs from
    /// the one its operands are stored in, they are bound as their `u32` words, a value each,
    /// which the kernel's stages decode at the read element; their arguments say so.
    pub fn served_kernel_elements(&self) -> [ElemType; 4] {
        [
            self.sum,
            self.elements.out,
            self.elements.query,
            self.elements.value,
        ]
    }

    /// The elements its operands may reach its two matrix instructions in, the cache as stored
    /// first. The score is read at the keys' element, the query converted to it as it is staged,
    /// as an FP8 cache's keys are multiplied on the FP8 instruction; and, where the reading allows
    /// it, at the query's, the keys widened to it. The value is read at the values' element, and,
    /// where the reading allows it, at the query's, the values widened to the element the model
    /// computes in. `f32` throughout is also rounded to tf32, ten bits of mantissa, the
    /// instruction an NVIDIA GPU offers `f32` operands, the sum staying `f32`.
    pub fn readings(&self) -> impl Iterator<Item = AttentionRead> {
        let AttentionOperands {
            query, key, value, ..
        } = self.elements;
        let converted = self.reading == AttentionReading::Rounded;
        let scores = [Some(key), (converted && query != key).then_some(query)];
        let values = [Some(value), (converted && query != value).then_some(query)];
        let f32 = ElemType::Float(FloatKind::F32);
        let tf32 = (converted
            && AttentionRead::all(f32) == AttentionRead { score: key, value }
            && query == f32)
            .then(|| AttentionRead::all(ElemType::Float(FloatKind::TF32)));
        let pairs: Vec<_> = scores
            .into_iter()
            .flatten()
            .flat_map(|score| {
                values
                    .into_iter()
                    .flatten()
                    .map(move |value| AttentionRead { score, value })
            })
            .collect();
        pairs.into_iter().chain(tf32)
    }

    /// This work item with its query and keys read in `read.score` and its values in
    /// `read.value`; the output stays as it is stored.
    pub fn read_as(&self, read: AttentionRead) -> AttentionWorkItem {
        AttentionWorkItem {
            elements: AttentionOperands {
                query: read.score,
                key: read.score,
                value: read.value,
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
            reading: MatmulReading::AsStored,
        }
    }

    /// The value, `probabilities · value`, as a shared multiplication microkernel is handed it:
    /// rows `Q` by columns `DV`, contracting `S`, out of probabilities of `probabilities` elements
    /// its owner keeps. The values lie contiguous along `DV`, the columns, as a row weight does.
    pub fn value(&self, probabilities: ElemType) -> MatmulWorkItem {
        MatmulWorkItem {
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
            reading: MatmulReading::AsStored,
        }
    }

    /// One stage of `keys` keys and their values, each over its own axes and bound innermost
    /// along the axis its lines run along, the keys in `key_order`.
    pub fn stage(&self, keys: usize, key_order: BindingOrder) -> [StagedOperand; 2] {
        let operand = |operand: AttentionOperand, row: Axis, extent: usize| {
            StagedOperand::new(
                self.elements.get(operand),
                Space::new(&[(S, keys), (row, extent)]),
                operand.innermost(key_order),
            )
        };
        [
            operand(AttentionOperand::Key, D, self.head_dim),
            operand(AttentionOperand::Value, DV, self.val_dim),
        ]
    }

    /// The query as a cube stages it once for its walk: one batch and one head of keys, `queries`
    /// rows of each of `heads` of its heads, over the whole head dim, its lines along it.
    pub fn staged_query(&self, heads: usize, queries: usize) -> StagedOperand {
        StagedOperand::new(
            self.elements.get(AttentionOperand::Query),
            Space::new(&[
                (B, 1),
                (KV, 1),
                (G, heads),
                (Q, queries),
                (D, self.head_dim),
            ]),
            D,
        )
    }
}
