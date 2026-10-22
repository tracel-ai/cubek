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

use crate::kernels::MatmulRoutineProblem;
use crate::kernels::select::problem::OperandKind;
use crate::microkernel::{MatmulLayout, MatmulOperand, MatmulOperands};
use cubecl::ir::ElemType;
use cubecl::tile::Axis;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_std::autotune::CandidatePriority;
use serde::{Deserialize, Serialize};

use cubecl::zspace::metadata::Metadata;

use cubek_std::layout::contiguous;

use cubek_std::layout;
use cubek_std::matmul_axes::K;

/// One input as the key tells problems apart: what it is, how it lies, and whether its batches
/// read one matrix.
///
/// The scheme is part of the identity because it decides what each candidate actually reads: a
/// q4 and a q8 weight of the same shape are different problems with different winners. So is the
/// layout: two problems of identical shape whose operands are laid out differently are different
/// problems, and sharing a record between them would hand one of them the other's winner. The
/// production weight is `[n, k]` behind a `[k, n]` view — contiguous along `K` — and its artifact
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
/// everywhere: `Linear` folds before projecting.
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
    /// The element the product is written in, the activation's or the one a packed activation
    /// stands for: it decides the vector widths, whether a gemm routine can reach a tensor-core
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
    /// in which kernels serve them, a shared weight read once for every batch, a batched one once
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
/// of batched activations and a product of batched operands count the same batches, and only the
/// first can fold them into rows of one weight.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MatmulBatching {
    /// Every batch reads its one matrix, so the batches are more rows against it.
    Shared,
    /// It holds a matrix per batch.
    Batched,
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

/// What a product is, as one candidate's cube meets it: whether a **candidate's
/// cuts are cuts this problem affords**.
///
/// Not a property of the problem alone — a block that tiles one product masks
/// another — and every arm is arithmetic over the extents and the block. How
/// much a masked cube or a lone one *costs* is a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulKind {
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
    /// [`Tiled`](MatmulKind::Tiled) candidates go first. A
    /// [`Masked`](MatmulKind::Masked) candidate falls one priority level per doubling of
    /// what its grid covers against what the problem asks for. A
    /// [`Whole`](MatmulKind::Whole) candidate goes last: a grid that cannot
    /// spread is the one claim about a launch that needs no clock — and even
    /// that says only where to try it, since how much it costs is still a
    /// measurement.
    pub fn priority(&self, block: CubeBlock) -> CandidatePriority {
        match self.kind(block) {
            MatmulKind::Tiled => CandidatePriority::FIRST,
            MatmulKind::Masked => CandidatePriority::FIRST
                .lowered((self.covered(block) / self.work().max(1)).max(1).ilog2()),
            MatmulKind::Whole => CandidatePriority::LAST,
        }
    }

    /// What this product is as a cube of `block` meets it. The arms are
    /// ordered by how much they take away and the first that holds is the kind:
    /// a block covering the product has no second cube to be masked against.
    pub fn kind(&self, block: CubeBlock) -> MatmulKind {
        let (block_m, block_n) = (block.m.max(1), block.n.max(1));
        if block_m >= self.m && block_n >= self.n {
            return MatmulKind::Whole;
        }
        match self.m.is_multiple_of(block_m) && self.n.is_multiple_of(block_n) {
            true => MatmulKind::Tiled,
            false => MatmulKind::Masked,
        }
    }
}

/// The problem a tuning **key** describes: dense operands of the key's element, the batch held by
/// the output, and by each input unless the key says it is shared, the lhs and the output
/// row-major and the rhs contiguous along the axis the key names. A col weight, an `[n, k]` buffer
/// behind a `[k, n]` view, is read where it lies
/// ([`MatmulTensors`](crate::kernels::MatmulTensors)), so its widths are its own; a rhs the key
/// names no axis for, a strided view, is aligned to one element.
impl From<&MatmulShape> for MatmulRoutineProblem {
    fn from(shape: &MatmulShape) -> Self {
        let rhs_layout = match shape.rhs.layout.axis() {
            Some(axis) if axis == K => MatmulLayout::Col,
            Some(_) => MatmulLayout::Row,
            None => MatmulLayout::Strided,
        };
        let held = |batching| match batching {
            MatmulBatching::Shared => 1,
            MatmulBatching::Batched => shape.batch,
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
            ..Self::dense(shape.m, shape.n, shape.k, shape.elem, rhs_layout)
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
    let m = lhs_shape[rank - 2];
    let k = lhs_shape[rank - 1];
    let n = rhs_shape[rhs_shape.rank() - 1];
    // The broadcast batch volume — one number, since the tuner only needs to
    // know how much work rides the leading axes, not how it is spread.
    let batches: usize = lhs_shape.as_slice()[..rank - 2]
        .iter()
        .zip(rhs_shape.as_slice()[..rank - 2].iter())
        .map(|(l, r)| *l.max(r))
        .product();
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

/// [`create_key`] keyed by the call's own rows, not the power of two above them: the key of a
/// table whose candidates serve or decline by the exact row count, so a winner measured at its
/// key's rows serves every call under it.
///
/// The few-rows tables' are such candidates: a plane shuffle lays its units out by the rows, and
/// what is left to the units along `K` follows, so a setting that serves eight rows may decline
/// five: against a float weight the vectors its units read along `K` narrow below two, and against
/// a packed one the turns a cube's planes take along `K` fall below two. A few rows is all those
/// tables take, so keying each count adds few records.
pub fn create_exact_rows_key(
    elem: ElemType,
    lhs: (&Metadata, OperandKind),
    rhs: (&Metadata, OperandKind),
) -> MatmulTuneKey {
    let shape = lhs
        .0
        .logical_shape()
        .expect("a tensor's tiling describes its own rank");
    MatmulTuneKey {
        m: shape[shape.rank() - 2].max(1),
        ..create_key(elem, lhs, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;
    use cubecl::quant::scheme::QuantScheme;

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

    /// A key by exact rows keeps the call's rows where the anchored key keeps the power of two
    /// above them, and is the anchored key in every other part.
    #[test]
    fn an_exact_rows_key_keeps_the_calls_rows() {
        let (k, n) = (256, 64);
        let rhs = Metadata::new([k, n], vec![1, k]);
        for (m, anchored) in [(1, 1), (5, 8), (13, 16), (32, 32)] {
            let lhs = Metadata::new([m, k], vec![k, 1]);
            let inputs = || ((&lhs, OperandKind::Float), (&rhs, OperandKind::Float));
            let exact = create_exact_rows_key(F16, inputs().0, inputs().1);
            let key = create_key(F16, inputs().0, inputs().1);
            assert_eq!(exact.m, m);
            assert_eq!(key.m, anchored);
            assert_eq!(
                MatmulTuneKey {
                    m: anchored,
                    ..exact
                },
                key
            );
        }
    }

    /// A weight broadcast along the batch dims is shared, however many batches the activation
    /// holds; one holding a matrix per batch is batched.
    #[test]
    fn a_weight_is_shared_where_its_batch_dims_hold_one_matrix() {
        assert_eq!(MatmulBatching::new(&[]), MatmulBatching::Shared);
        assert_eq!(MatmulBatching::new(&[1, 1]), MatmulBatching::Shared);
        assert_eq!(MatmulBatching::new(&[4]), MatmulBatching::Batched);
        assert_eq!(MatmulBatching::new(&[1, 8]), MatmulBatching::Batched);
    }

    /// The keys of a decode projection against a float col weight and against a q4 weight
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
