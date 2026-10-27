//! How a product is told apart from another, for the tables — and what a table
//! reads off it when it prioritizes its candidates.
//!
//! The key is a description of the *problem* — what the operands are, how big
//! they are, how they are laid out — so it lives beside the plans rather than
//! beside the tables that consume it. A caller builds the tables; it is the
//! kernels that know what makes two products the same product, and the fused matmul asks the many-rows table's tuner for its election
//! under this key too.
//!
//! [`MatmulShape`] is the second half: the key read as extents, which is what a
//! candidate list is prioritized against. Both arms prioritize against it, and
//! it reads the key alone because the key is the whole of what cubecl hands a priority
//! function.

use crate::kernels::select::problem::{MatmulProblem, OperandKind};
use crate::kernels::{MatmulIndex, MatmulRoutineProblem};
use cubecl::ir::ElemType;
use cubecl::tile::Axis;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_microkernel::matmul::{MatmulLayout, MatmulOperand, MatmulOperands};
use cubek_std::autotune::CandidatePriority;
use serde::{Deserialize, Serialize};

use cubecl::zspace::metadata::Metadata;

use cubek_std::layout::contiguous;

use cubek_std::layout;

/// One input as the key tells problems apart: what it is, how it lies, and whether its batches
/// read one matrix.
///
/// The scheme is part of the identity because it decides what each candidate actually reads: a
/// q4 and a q8 input of the same shape are different problems with different winners. So is the
/// layout: two problems of identical shape whose operands are laid out differently are different
/// problems, and sharing a record between them would hand one of them the other's winner. The
/// production rhs is `[n, k]` behind a `[k, n]` view — contiguous along `K` — and its artifact
/// form is contiguous along `N`; the plane shuffle reads the two in opposite orders. The
/// backward's products arrive with the left-hand side transposed, which the engine serves by its
/// strides, and a strided or broadcast view is a third thing again: one the kernels stage through
/// a copy or decline.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OperandKey {
    pub kind: OperandKind,
    pub layout: OperandLayout,
    pub batching: MatmulBatching,
}

/// What makes two products want the same kernel.
///
/// `k` and `n` are **exact**: they are a projection's dimensions, fixed by the
/// model config, so the key space is that architecture's handful of shapes and
/// anchoring would merge problems whose winners differ. `m` is anchored — it is
/// the axis a caller varies freely (batch × sequence), and it is the axis the
/// crossovers actually live on, so its buckets are where candidates trade
/// places. `batches` is anchored for the same reason and is 1 almost
/// everywhere: `Linear` merges before projecting.
///
/// **The family anchors, not the derive.** cubecl's anchor attribute scales
/// its base by the autotune level, and that level is set for how close to
/// the roofline a round may stop;
/// under it a 64-row product would key as 87. A candidate list prioritizes its
/// candidates and proves its duplicates from the key's `m`
/// ([`TuneCandidate::blueprint`](cubek_std::autotune::TuneCandidate::blueprint)), so
/// the bucket's representative has to be a shape the derivation tiles the way
/// it tiles the shapes inside the bucket — a power of two, at every level.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MatmulTuneKey {
    /// The element the product is written in, the lhs's or the one a packed lhs
    /// stands for: it decides the vector widths, whether a staged routine can reach a tensor-core
    /// leaf, and what the memory-bound kernels accumulate from.
    elem: ElemType,
    /// Reduction length.
    k: usize,
    /// Output width.
    pub n: usize,
    /// Rows, anchored to the power of two above them — see `anchored`.
    pub m: usize,
    /// Broadcast batches, anchored the same way. Whether each input's batches read one matrix
    /// is its own ([`OperandKey`]): two products of one shape differ there in what they move and
    /// in which kernels serve them, a shared input read once for every batch, a batched one once
    /// per batch.
    batches: usize,
    lhs: OperandKey,
    rhs: OperandKey,
}

/// How an operand lies in its buffer, as a key tells two products apart: two of one shape laid
/// out differently are read differently, and want different winners.
///
/// An autotune record is named by the key's [`Debug`], and `Axis`'s own prints the index it wraps,
/// `Axis(0)` where a reader wants `M`. So a layout prints the same letter
/// [`axis_name`](cubek_std::matmul_axes::axis_name) gives the printed space.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum OperandLayout {
    /// Contiguous along this axis: a row or a col view.
    Along(Axis),
    /// Contiguous along neither axis: a strided or broadcast view, read one element at a time.
    Strided,
    /// Contiguous along this axis inside tiles of these extents over the operand's two axes: a
    /// buffer stored a tile at a time. One of one value a tile is [`Along`](Self::Along).
    Tiles(Axis, (usize, usize)),
}

impl OperandLayout {
    /// Contiguous along `axis`, or strided where there is none.
    pub fn along(axis: Option<Axis>) -> Self {
        axis.map_or(Self::Strided, Self::Along)
    }

    /// The axis the operand is read along: the one it is contiguous along, or the one its stored
    /// pieces run along; `None` for a strided one.
    pub fn axis(self) -> Option<Axis> {
        match self {
            Self::Along(axis) | Self::Tiles(axis, _) => Some(axis),
            Self::Strided => None,
        }
    }
}

impl core::fmt::Debug for OperandLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Along(axis) => f.write_str(cubek_std::matmul_axes::axis_name(*axis)),
            Self::Strided => f.write_str("strided"),
            Self::Tiles(axis, (first, second)) => write!(
                f,
                "{} in {first}x{second} tiles",
                cubek_std::matmul_axes::axis_name(*axis)
            ),
        }
    }
}

impl MatmulTuneKey {
    /// Whether either input is packed: what a candidate that reads packed inputs as stored serves,
    /// and one that reads float inputs declines, since a shape alone does not say how an input is
    /// stored.
    pub fn is_packed(&self) -> bool {
        [self.lhs.kind, self.rhs.kind]
            .iter()
            .any(|kind| matches!(kind, OperandKind::Quantized(_)))
    }

    /// The problem this key describes, as a candidate list prioritizes against it.
    pub fn shape(&self) -> MatmulShape {
        MatmulShape {
            m: self.m,
            n: self.n,
            k: self.k,
            batch: self.batches.max(1),
            elem: self.elem,
            lhs: self.lhs,
            rhs: self.rhs,
        }
    }

    /// The power of two at or above `extent`, capped at `max`: the bucket a free
    /// axis is keyed by, so a record stands for the range under its representative
    /// and the representative is a shape every block tiles.
    pub fn anchored(extent: usize, max: usize) -> usize {
        extent.next_power_of_two().clamp(1, max)
    }
}

/// The problem as a candidate list prioritizes a candidate against it: the
/// extents, the element, and each input as the key states it.
///
/// Read off the key and nothing else, because two problems sharing a key share
/// a tuning record — a priority that separated them would plan one round and
/// persist it for both. So `m` and `batch` are the key's *anchored* values, one
/// bucket standing for the range around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Broadcast batches, `1` when unbatched. Work the grid carries **whole**:
    /// a batch axis is never spent on planes, so it multiplies the cubes a
    /// block leaves rather than changing their shape.
    pub batch: usize,
    pub elem: ElemType,
    pub lhs: OperandKey,
    /// The rhs, the axis it is contiguous along deciding which axis a memory-bound kernel's
    /// instances follow.
    pub rhs: OperandKey,
}

impl OperandKey {
    /// A float input contiguous along `axis`, strided where it names none, batched as `batching`
    /// says.
    pub fn float(axis: Option<Axis>, batching: MatmulBatching) -> Self {
        Self {
            kind: OperandKind::Float,
            layout: OperandLayout::along(axis),
            batching,
        }
    }
}

/// Whether an operand's batches read one matrix. A batch count alone does not say: a projection
/// of a batched lhs and a product of batched operands count the same batches, and only the
/// first can merge them into rows against one rhs.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MatmulBatching {
    /// Every batch reads its one matrix, so the batches are more rows against it.
    Shared,
    /// It holds a matrix per batch.
    Batched,
    /// It is a stack of `slabs` matrices, and each batch reads the one its entry of the product's
    /// table names ([`MatmulIndex::Slab`]).
    Slabs { slabs: usize },
}

impl MatmulBatching {
    /// The batching of an operand whose leading dims are `batch_dims`: shared where they hold one
    /// matrix, as an operand broadcast along them does.
    pub fn new(batch_dims: &[usize]) -> Self {
        match batch_dims.iter().product::<usize>() {
            1 => Self::Shared,
            _ => Self::Batched,
        }
    }
}

/// What one cube of a candidate covers: its output block, and how deep one
/// step of its `K` walk runs. The currency every priority statement is arithmetic
/// over, and the one thing a candidate has to say about itself before a problem
/// arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeBlock {
    pub m: usize,
    pub n: usize,
    /// Elements of the reduction one step of the walk contracts. The whole of
    /// `k` where the walk is one pass.
    pub k: usize,
}

/// How a grid of one candidate's cubes covers a product: whether a **candidate's
/// cuts are cuts this problem affords**.
///
/// Not a property of the problem alone — a block that tiles one product masks
/// another — and every arm is arithmetic over the extents and the block. How
/// much a masked cube or a lone one *costs* is a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridCoverage {
    /// One cube covers the whole product: the grid is the batch and nothing
    /// else, so no device spreads this candidate over more than one worker per
    /// batch.
    Whole,
    /// The block hangs off an output axis, and the grid pays for the share of
    /// every cube that falls outside the problem.
    Masked,
    /// The block tiles both output axes: every cube it distributes is full.
    Tiled,
}

impl MatmulShape {
    /// Output elements the product covers — the work a grid is cut from.
    pub fn work(&self) -> usize {
        self.m * self.n * self.batch
    }

    /// Cubes a grid of `block` holds: whole blocks along each output axis,
    /// partial ones counted, and the batch riding on top.
    pub fn cubes(&self, block: CubeBlock) -> usize {
        self.m.div_ceil(block.m.max(1)) * self.n.div_ceil(block.n.max(1)) * self.batch
    }

    /// Output elements that grid **covers**, masking included — what the
    /// device pays for, against [`work`](Self::work), which is what it gets.
    pub fn covered(&self, block: CubeBlock) -> usize {
        self.cubes(block) * block.m.max(1) * block.n.max(1)
    }

    /// Where a candidate whose cube covers `block` is tried on this problem —
    /// the priority both arms are written in.
    ///
    /// [`Tiled`](GridCoverage::Tiled) candidates go first. A
    /// [`Masked`](GridCoverage::Masked) candidate falls one priority level per doubling of
    /// what its grid covers against what the problem asks for. A
    /// [`Whole`](GridCoverage::Whole) candidate goes last: a grid that cannot
    /// spread is the one claim about a launch that needs no clock — and even
    /// that says only where to try it, since how much it costs is still a
    /// measurement.
    pub fn priority(&self, block: CubeBlock) -> CandidatePriority {
        match self.coverage(block) {
            GridCoverage::Tiled => CandidatePriority::FIRST,
            GridCoverage::Masked => CandidatePriority::FIRST
                .lowered((self.covered(block) / self.work().max(1)).max(1).ilog2()),
            GridCoverage::Whole => CandidatePriority::LAST,
        }
    }

    /// How a grid of `block` cubes covers this product. The arms are
    /// ordered by how much they take away and the first that holds is the coverage:
    /// a block covering the product has no second cube to be masked against.
    pub fn coverage(&self, block: CubeBlock) -> GridCoverage {
        let (block_m, block_n) = (block.m.max(1), block.n.max(1));
        if block_m >= self.m && block_n >= self.n {
            return GridCoverage::Whole;
        }
        match self.m.is_multiple_of(block_m) && self.n.is_multiple_of(block_n) {
            true => GridCoverage::Tiled,
            false => GridCoverage::Masked,
        }
    }
}

/// The problem a tuning **key** describes: dense operands of the key's element, the batch held by
/// the output, and by each input unless the key says it is shared, the output row-major and each
/// input contiguous along the axis the key names. A col input, its buffer the view's transpose,
/// is read where it lies ([`MatmulTensors`](crate::kernels::MatmulTensors)), so its widths are
/// its own; an input the key names no axis for, a strided view, is aligned to one element.
impl From<&MatmulShape> for MatmulRoutineProblem {
    fn from(shape: &MatmulShape) -> Self {
        let held = |batching| match batching {
            MatmulBatching::Shared | MatmulBatching::Slabs { .. } => 1,
            MatmulBatching::Batched => shape.batch,
        };
        let index = match shape.rhs.batching {
            MatmulBatching::Slabs { slabs } => Some(MatmulIndex::Slab { slabs }),
            MatmulBatching::Shared | MatmulBatching::Batched => None,
        };
        Self {
            batches: (shape.batch > 1)
                .then_some(shape.batch)
                .into_iter()
                .collect(),
            batches_held: MatmulOperands {
                lhs: held(shape.lhs.batching),
                rhs: held(shape.rhs.batching),
                out: shape.batch,
            },
            index,
            ..Self::dense(shape.m, shape.n, shape.k, shape.elem)
                .laid(
                    MatmulOperand::Lhs,
                    MatmulLayout::along(shape.lhs.layout.axis(), MatmulOperand::Lhs),
                )
                .laid(
                    MatmulOperand::Rhs,
                    MatmulLayout::along(shape.rhs.layout.axis(), MatmulOperand::Rhs),
                )
        }
    }
}

/// The key of `lhs · rhs` in `elem`, each input its metadata, read as its logical shape, and what
/// it is.
pub fn create_key(
    elem: ElemType,
    (lhs_meta, lhs_kind): (&Metadata, OperandKind),
    (rhs_meta, rhs_kind): (&Metadata, OperandKind),
) -> MatmulTuneKey {
    let logical = |meta: &Metadata| {
        meta.logical_shape()
            .expect("a tensor's tiling describes its own rank")
    };
    let (lhs_shape, rhs_shape) = (logical(lhs_meta), logical(rhs_meta));
    let rank = lhs_shape.rank();
    debug_assert_eq!(
        rank,
        rhs_shape.rank(),
        "the batch dims are read pairwise off operands of one rank"
    );
    // The broadcast batch volume is one number, since the tuner only needs to know how much work
    // rides the leading axes, not how it is spread.
    let MatmulProblem {
        m, k, n, batches, ..
    } = MatmulProblem::new((&lhs_shape, lhs_kind), (&rhs_shape, rhs_kind));
    // An operand keys by the axis it is contiguous along and the tile it is stored in, so a
    // winner elected on one layout is never served to another, which its kernel may decline.
    let operand = |meta: &Metadata, kind: OperandKind, operand: MatmulOperand| {
        let shape = logical(meta);
        OperandKey {
            kind,
            layout: match (contiguous(meta, operand.axes()), layout::tile(meta)) {
                (None, _) => OperandLayout::Strided,
                (Some(axis), (1, 1)) => OperandLayout::Along(axis),
                (Some(axis), tile) => OperandLayout::Tiles(axis, tile),
            },
            batching: MatmulBatching::new(&shape.as_slice()[..shape.rank() - 2]),
        }
    };

    MatmulTuneKey::new(
        elem,
        k,
        n,
        MatmulTuneKey::anchored(m, 8192),
        MatmulTuneKey::anchored(batches, 64),
        operand(lhs_meta, lhs_kind, MatmulOperand::Lhs),
        operand(rhs_meta, rhs_kind, MatmulOperand::Rhs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;
    use cubecl::quant::scheme::QuantScheme;
    use cubek_std::matmul_axes::K;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// A free axis keys by the power of two above it, whatever the autotune
    /// level: the priority and the plan-time proof read the key's `m`, and a
    /// bucket represented by 87 rows would derive blocks nothing at 64 rows
    /// runs.
    #[test]
    fn a_free_axis_keys_by_the_power_of_two_above_it() {
        assert_eq!(MatmulTuneKey::anchored(1, 8192), 1);
        assert_eq!(MatmulTuneKey::anchored(64, 8192), 64);
        assert_eq!(MatmulTuneKey::anchored(65, 8192), 128);
        assert_eq!(MatmulTuneKey::anchored(100_000, 8192), 8192);
        assert_eq!(MatmulTuneKey::anchored(0, 64), 1);
    }

    /// An input broadcast along the batch dims is shared, however many batches the other one
    /// holds; one holding a matrix per batch is batched.
    #[test]
    fn an_input_is_shared_where_its_batch_dims_hold_one_matrix() {
        assert_eq!(MatmulBatching::new(&[]), MatmulBatching::Shared);
        assert_eq!(MatmulBatching::new(&[1, 1]), MatmulBatching::Shared);
        assert_eq!(MatmulBatching::new(&[4]), MatmulBatching::Batched);
        assert_eq!(MatmulBatching::new(&[1, 8]), MatmulBatching::Batched);
    }

    /// A key whose rhs is a stack of slabs states a problem every batch of which reads one slab
    /// through the table: the stack is held once, not once a batch.
    #[test]
    fn a_stacked_rhs_states_a_problem_read_through_the_slab_table() {
        let slabs = MatmulBatching::Slabs { slabs: 8 };
        let key = MatmulTuneKey::new(
            F16,
            256,
            64,
            1,
            32,
            OperandKey::float(Some(K), MatmulBatching::Batched),
            OperandKey::float(Some(K), slabs),
        );
        let problem = MatmulRoutineProblem::from(&key.shape());
        assert_eq!(problem.index, Some(MatmulIndex::Slab { slabs: 8 }));
        assert_eq!(problem.batches, vec![32]);
        assert_eq!(problem.batches_held.lhs, 32);
        assert_eq!(problem.batches_held.rhs, 1);
    }

    /// The keys of a decode projection against a float col rhs and against a q4 rhs
    /// packed along `k`, as they stand: a key is what a saved table is looked up by, so a change
    /// that moves one re-tunes every machine, and is made on purpose or not at all.
    #[test]
    fn a_projections_keys_are_pinned() {
        use cubecl::quant::scheme::{QuantStore, QuantValue, ScaleDtype};
        let (m, k, n) = (4, 256, 64);
        let lhs = Metadata::new([m, k], vec![k, 1]);
        let col = Metadata::new([k, n], vec![1, k]);
        let q4 = QuantScheme::default()
            .with_value(QuantValue::Q4S)
            .per_block(vec![32, 1], ScaleDtype::F16)
            .with_store(QuantStore::PackedU32(1));
        let packed = Metadata::new([k, n], vec![1, k / 8]);
        let key = |rhs: &Metadata, kind| {
            format!(
                "{:?}",
                create_key(F16, (&lhs, OperandKind::Float), (rhs, kind))
            )
        };
        assert_eq!(
            key(&col, OperandKind::Float),
            "MatmulTuneKey { elem: Float(F16), k: 256, n: 64, m: 4, batches: 1, lhs: OperandKey { kind: Float, layout: K, batching: Shared }, rhs: OperandKey { kind: Float, layout: K, batching: Shared } }"
        );
        assert_eq!(
            key(&packed, OperandKind::Quantized(q4)),
            "MatmulTuneKey { elem: Float(F16), k: 256, n: 64, m: 4, batches: 1, lhs: OperandKey { kind: Float, layout: K, batching: Shared }, rhs: OperandKey { kind: Quantized(QuantScheme { value: Q4S, store: PackedU32(1), mode: Symmetric, tensor: None, block: Some(BlockScale { size: BlockSize([32, 1]), dtype: F16 }) }), layout: K, batching: Shared } }"
        );
    }
}
