//! The problem a routine computes, and the axis its queries are read along.

use crate::kernels::{AttentionVisibility, PackedLayout};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Space;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{
    AttentionOperand, AttentionOperands, AttentionReading, AttentionWorkItem, B, D, DV, G, KV, Q, S,
};
use cubek_std::routine::EstimatedProblem;

/// `out[b, q, h] = softmax(query[b, h, q] · key[b, h / group]ᵀ · scale) · value[b, h / group]`,
/// as the host describes it, its queries along the axis [`QueryAxis`] says: a single position's
/// `group` heads are `group` queries of one head.
///
/// The queries are the last `queries` positions of the attended keys, whose count `len` only the
/// device holds: `keys` is the capacity the key and value buffers are allocated to, of which the
/// kernel reads the first `len`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionRoutineProblem {
    pub batch: usize,
    pub kv_heads: usize,
    /// Query heads per key head.
    pub group: usize,
    pub queries: usize,
    /// The keys the buffers hold, attended or not.
    pub keys: usize,
    pub head: usize,
    pub value: usize,
    /// Which of the attended keys a query sees.
    pub visibility: AttentionVisibility,
    /// The element each operand is stored in.
    pub elements: AttentionOperands<ElemType>,
    /// The alignment of each operand's buffer in elements, over the axes this kind names.
    pub elem_alignments: AttentionOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: AttentionOperands<u64>,
    /// How the keys and values lie.
    pub cache: AttentionCacheLayout,
}

/// How an attention's keys and values lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionCacheLayout {
    /// As sequences, `[b, kv, s, ·]`.
    Sequences,
    /// In two pools of pages of `page_size` positions, each sequence's pages named in order by a
    /// table.
    Pages { page_size: usize },
    /// Packed into words of codebook indices, as the layout states.
    Packed(PackedLayout),
}

impl AttentionRoutineProblem {
    /// The element every microkernel of this kind sums in, whatever it reads and writes.
    pub const SUM: ElemType = ElemType::Float(FloatKind::F32);

    /// The widest capacity [`capacity_classes`](Self::capacity_classes) lists: a multiple of
    /// every step of keys a routine walks.
    pub const WIDEST_CAPACITY_CLASS: usize = 2048;

    /// A multiple of every step of keys a routine walks, at every setting it lists: a cache
    /// holding a multiple of it binds its keys as whole steps, so no launch over it masks its
    /// reads. A capacity off it compiles the bounded launch, which masks every read of its walk
    /// rather than its last step's alone.
    pub const WIDEST_STEP: usize = 256;

    /// The capacities, up to `limit`, a caller compiling ahead of time runs a prefill at: every
    /// power of two from 2 to [`WIDEST_CAPACITY_CLASS`](Self::WIDEST_CAPACITY_CLASS), the last
    /// one inside `limit` standing for it.
    ///
    /// A routine's launch compiles apart where the capacity is not a multiple of its step of
    /// keys, the walk's last step then bounded. Two is a multiple of no step wider than it, and
    /// the widest class of every step, so a walk over the classes compiles both launches of every
    /// setting; the powers between land the tuning key's buckets ([`AttentionKey`](crate::kernels::tune::key::AttentionKey)).
    pub fn capacity_classes(limit: usize) -> Vec<usize> {
        let mut classes = Vec::new();
        for class in std::iter::successors(Some(2usize), |class| Some(class * 2)) {
            classes.push(class.min(limit));
            if class >= limit || class >= Self::WIDEST_CAPACITY_CLASS {
                break;
            }
        }
        classes
    }

    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> AttentionWorkItem {
        AttentionWorkItem {
            elements: self.elements,
            sum: Self::SUM,
            reading: match self.cache {
                AttentionCacheLayout::Pages { .. } => AttentionReading::AsStored,
                AttentionCacheLayout::Sequences | AttentionCacheLayout::Packed(_) => {
                    AttentionReading::Rounded
                }
            },
            page_size: match self.cache {
                AttentionCacheLayout::Pages { page_size } => Some(page_size),
                AttentionCacheLayout::Sequences | AttentionCacheLayout::Packed(_) => None,
            },
            group: self.group,
            head: self.head,
            value: self.value,
        }
    }

    /// Query heads.
    pub fn heads(&self) -> usize {
        self.kv_heads * self.group
    }

    /// Every axis of the problem at its extent: what a routine partitions.
    pub fn space(&self) -> Space {
        Space::new(&[
            (B, self.batch),
            (KV, self.kv_heads),
            (G, self.group),
            (Q, self.queries),
            (S, self.keys),
            (D, self.head),
            (DV, self.value),
        ])
    }

    /// A problem of dense operands of `element` throughout, their views starting where their
    /// buffers do: what a routine's pure tests prepare against.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense(
        (batch, heads, kv_heads): (usize, usize, usize),
        (queries, keys): (usize, usize),
        (head, value): (usize, usize),
        element: ElemType,
    ) -> Self {
        let aligned = |innermost: usize| ElemAlignment::new(&[1, innermost], &[innermost, 1]);
        Self {
            batch,
            kv_heads,
            group: heads / kv_heads,
            queries,
            keys,
            head,
            value,
            visibility: AttentionVisibility::Attended,
            elements: AttentionOperands {
                query: element,
                key: element,
                value: element,
                out: element,
            },
            elem_alignments: AttentionOperands {
                query: aligned(head),
                key: aligned(head),
                value: aligned(value),
                // The output's heads are adjacent, so a query head's row starts a row apart.
                out: aligned(value),
            },
            start_bytes: AttentionOperands {
                query: 0,
                key: 0,
                value: 0,
                out: 0,
            },
            cache: AttentionCacheLayout::Sequences,
        }
    }

    /// Elements `operand` holds.
    pub fn elements_of(&self, operand: AttentionOperand) -> usize {
        let (queries, keys) = (
            self.batch * self.heads() * self.queries,
            self.batch * self.kv_heads * self.keys,
        );
        match operand {
            AttentionOperand::Query => queries * self.head,
            AttentionOperand::Key => keys * self.head,
            AttentionOperand::Value => keys * self.value,
            AttentionOperand::Out => queries * self.value,
        }
    }
}

impl EstimatedProblem for AttentionRoutineProblem {
    /// Two operations per multiply-add of both contractions, every query against every key the
    /// buffers hold, and every operand's elements once. The attended keys are the device's: a
    /// problem is sized to its capacity, and one whose queries see fewer to all of it.
    fn demand(&self) -> Work {
        let pairs = self.batch * self.heads() * self.queries * self.keys;
        Work {
            compute_ops: 2 * pairs * (self.head + self.value),
            bytes: AttentionOperand::ALL
                .iter()
                .map(|&operand| self.elements_of(operand) * self.elements.get(operand).size())
                .sum(),
        }
    }
}

/// What a routine that reads float keys and values declines on a packed cache.
pub const ATTENTION_PACKED_UNREAD: Refusal = Refusal::Declined {
    why: "this attention routine reads float keys and values, not a packed cache",
};

/// What a problem's query axis `Q` walks: the query positions of one head, or the heads of one key
/// head's group at a single position.
///
/// A single position of a group of heads is the same attention as each head at that position: it
/// sees every attended key, or the last of them a window spans, and the heads of a group read the
/// same keys. Walked along `Q`, the group fills the part of a fragment one position would leave
/// empty, and its keys are walked once for all of it; the heads are read at their own stride, so
/// nothing is copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryAxis {
    /// The query positions of one head, the heads of a group beside them.
    Positions,
    /// A single position's group of heads.
    GroupHeads,
}

impl QueryAxis {
    /// The query axis of an attention of `queries` positions.
    pub fn new(queries: usize) -> Self {
        match queries {
            1 => Self::GroupHeads,
            _ => Self::Positions,
        }
    }

    /// An attention of `group` heads a key head and `queries` positions seeing the keys
    /// `visibility` says, as a problem along this axis states it: the heads beside each query,
    /// the queries, and which keys a query sees.
    pub fn read(
        self,
        group: usize,
        queries: usize,
        visibility: AttentionVisibility,
    ) -> (usize, usize, AttentionVisibility) {
        match self {
            Self::Positions => (group, queries, visibility),
            Self::GroupHeads => (1, group * queries, visibility.of_last_position()),
        }
    }

    /// `tensor`'s heads split into a group per key head over the same buffer, the queries along
    /// this axis: `[b, kv, g, q, x]`, or where it walks the group's heads `[b, kv, q, g, x]`, the
    /// single position and the heads trading places. `dims` names which of the tensor's four dims
    /// are the batch, the heads, the queries and the last.
    pub(crate) fn view(self, tensor: &PlainTensor, group: usize, dims: [usize; 4]) -> PlainTensor {
        let (shape, strides) = (&tensor.binding.shape, &tensor.binding.strides);
        let [batch, heads, queries, last] = dims;
        let (heads_beside, along_queries) = match self {
            Self::Positions => ((group, strides[heads]), (shape[queries], strides[queries])),
            Self::GroupHeads => ((shape[queries], strides[queries]), (group, strides[heads])),
        };
        let view_shape = vec![
            shape[batch],
            shape[heads] / group,
            heads_beside.0,
            along_queries.0,
            shape[last],
        ];
        let view_strides = vec![
            strides[batch],
            strides[heads] * group,
            heads_beside.1,
            along_queries.1,
            strides[last],
        ];
        tensor.viewed(Metadata::new(view_shape, view_strides))
    }
}
