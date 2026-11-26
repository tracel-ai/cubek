//! How a product is told apart from another, for a caller that measures the routines — and what
//! the key states of a product that a caller's candidate list is ordered against.
//!
//! The key is a description of the *problem* — what the operands are, how big they are, how they
//! are laid out — so it lives beside the plans rather than beside the autotunes that consume it. A
//! caller builds the autotunes; it is the kernels that know what makes two products the same
//! product, and the fused matmul reads the unfused autotune's election under this key too.
//!
//! [`MatmulShape`] is the second half: the key read as extents, and how a cube's output block
//! covers them ([`GridCoverage`]). It reads the key alone because the key is the whole of what
//! cubecl hands a priority function.

use crate::kernels::MatmulRoutineProblem;
use crate::kernels::select::problem::{MatmulProblem, OperandKind};
use cubecl::ir::ElemType;
use cubecl::tile::Dimension;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_microkernel::matmul::{
    MatmulDimensions, MatmulLayout, MatmulOperand, MatmulOperands, MatmulStoredTile,
};
use serde::{Deserialize, Serialize};

use cubecl::zspace::metadata::Metadata;

use cubek_std::layout::contiguous;

use cubecl::ir::UIntKind;
use cubecl::tile::layout::Stagger;
use cubek_std::Hardware;
use cubek_std::layout;
use cubek_std::layout::stored::{Operand, Packing};
use cubek_std::layout::tile_order::{TileOrder, TileWords};
use cubek_std::matmul_dimensions::{K, M, N};

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
/// the dimension a caller varies freely (batch × sequence), and it is the dimension the
/// crossovers actually live on, so its buckets are where candidates trade
/// places. `batches` is anchored for the same reason and is 1 almost
/// everywhere: `Linear` merges before projecting.
///
/// **The family anchors, not the derive.** cubecl's anchor attribute scales
/// its base by the autotune level, and that level is set for how close to
/// the roofline a round may stop;
/// under it a 64-row product would key as 87. A caller's candidate list prioritizes its
/// candidates and proves its duplicates from the launch each prepares for the key's `m`, so
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
/// An autotune record is named by the key's [`Debug`], and `Dimension`'s own prints the index it
/// wraps, `Dimension(0)` where a reader wants `M`. So a layout prints the same letter
/// [`dimension_name`](cubek_std::matmul_dimensions::dimension_name) gives the printed space.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum OperandLayout {
    /// Contiguous along this dimension: a row or a col view.
    Along(Dimension),
    /// Contiguous along neither dimension: a strided or broadcast view, read one element at a time.
    Strided,
    /// Contiguous along this dimension inside tiles of these extents over the operand's two
    /// dimensions, the tiles' words lying as the order says and their strips as the strips say: a
    /// buffer stored a tile at a time. One of one value a tile is [`Along`](Self::Along). A float
    /// buffer's tiles, and a packed one's laid row by row, are a row's run a tile row; a packed
    /// one's laid block by block are not, and a kernel may decline one where it reads the other.
    Tiles(Dimension, (usize, usize), TileWords, TileStrips),
}

/// How a packed operand's strips of tiles lie in its buffer ([`TileOrder::stagger`]): what a key
/// holds of the order beyond its tile and its words, so the problem a packed key describes is the
/// one its tensors state ([`MatmulTuneKey::packed_problem`]).
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TileStrips {
    /// Each strip right after the one before it.
    BackToBack,
    /// Each strip an odd number of memory transactions of `transaction_bits` after the one
    /// before it.
    Staggered {
        /// Bits one memory transaction moves.
        transaction_bits: usize,
    },
}

impl TileStrips {
    /// How the strips of a matrix stored in `order` lie.
    pub fn of(order: TileOrder) -> Self {
        order
            .stagger()
            .map_or(Self::BackToBack, |stagger| Self::Staggered {
                transaction_bits: stagger.transaction_bits(),
            })
    }

    /// `order` with its strips laid as these are, or `None` where a transaction holds no whole
    /// number of the order's values.
    pub fn laid(self, order: TileOrder) -> Option<TileOrder> {
        match self {
            Self::BackToBack => Some(order),
            Self::Staggered { transaction_bits } => {
                Some(order.staggered(Stagger::new(transaction_bits, u8::BITS as usize)?))
            }
        }
    }
}

impl OperandLayout {
    /// Contiguous along `dimension`, or strided where there is none.
    pub fn along(dimension: Option<Dimension>) -> Self {
        dimension.map_or(Self::Strided, Self::Along)
    }

    /// The dimension the operand is read along: the one it is contiguous along, or the one its
    /// stored pieces run along; `None` for a strided one.
    pub fn dimension(self) -> Option<Dimension> {
        match self {
            Self::Along(dimension) | Self::Tiles(dimension, ..) => Some(dimension),
            Self::Strided => None,
        }
    }
}

impl core::fmt::Debug for OperandLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Along(dimension) => {
                f.write_str(cubek_std::matmul_dimensions::dimension_name(*dimension))
            }
            Self::Strided => f.write_str("strided"),
            Self::Tiles(dimension, (first, second), words, strips) => {
                write!(
                    f,
                    "{} in {first}x{second} tiles",
                    cubek_std::matmul_dimensions::dimension_name(*dimension)
                )?;
                // Row by row and back to back are the tiles every key named before the order and
                // the strips did.
                match words {
                    TileWords::RowByRow => Ok(()),
                    TileWords::BlockByBlock => f.write_str(" block by block"),
                }?;
                match strips {
                    TileStrips::BackToBack => Ok(()),
                    TileStrips::Staggered { transaction_bits } => {
                        write!(f, " staggered by {}B", transaction_bits / 8)
                    }
                }
            }
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

    /// The problem a packed key describes on `hw`: the one the tensors it was read off state
    /// ([`MatmulTensors::problem`](crate::kernels::MatmulTensors::problem)), each packed input
    /// read as its scheme, its layout and its strips say, a float input dense in the key's
    /// element and laid as the key says, the batches merged into the rows. `None` for a key with
    /// no packed input, a packed input no read takes as stated, or a float input stored in tiles
    /// ([`OperandKey::float_layout`]).
    pub fn packed_problem(&self, hw: &Hardware) -> Option<MatmulRoutineProblem> {
        if !self.is_packed() {
            return None;
        }
        let packings = MatmulOperands {
            lhs: self.lhs.packing([M, K], hw)?,
            rhs: self.rhs.packing([K, N], hw)?,
            out: None,
        };
        let stored_tile = |packing: Option<Packing>| {
            let order = packing?.operand.tile_order()?;
            let (columns, rows) = order.tile();
            (order.words() == TileWords::RowByRow).then_some(MatmulStoredTile { rows, columns })
        };
        let element = |packing: Option<Packing>| match packing {
            Some(_) => MatmulRoutineProblem::PACKED,
            None => self.elem,
        };
        let lay =
            |problem: MatmulRoutineProblem, input: MatmulOperand, key: OperandKey| match packings
                .get(input)
            {
                Some(_) => Some(problem),
                None => Some(problem.laid(input, key.float_layout(input)?)),
            };
        let problem = MatmulRoutineProblem {
            stored_tiles: MatmulOperands {
                lhs: stored_tile(packings.lhs),
                rhs: stored_tile(packings.rhs),
                out: None,
            },
            elements: MatmulOperands {
                lhs: element(packings.lhs),
                rhs: element(packings.rhs),
                out: self.elem,
            },
            ..MatmulRoutineProblem::packed(
                self.m * self.batches.max(1),
                self.n,
                self.k,
                self.elem,
                packings,
            )
        };
        let problem = lay(problem, MatmulOperand::Lhs, self.lhs)?;
        lay(problem, MatmulOperand::Rhs, self.rhs)
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
    /// dimension is keyed by, so a record stands for the range under its representative
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
    /// a batch dimension is never spent on planes, so it multiplies the cubes a
    /// block leaves rather than changing their shape.
    pub batch: usize,
    pub elem: ElemType,
    pub lhs: OperandKey,
    /// The rhs, the dimension it is contiguous along deciding which dimension a memory-bound
    /// kernel's instances follow.
    pub rhs: OperandKey,
}

impl OperandKey {
    /// How this input is stored, over `dimensions` as the product names them (`[M, K]` for the
    /// lhs, `[K, N]` for the rhs), read `hw`'s loads at a time: `Some(None)` for a float input,
    /// `None` for a packed one no read takes as the key states it.
    pub fn packing(self, dimensions: [Dimension; 2], hw: &Hardware) -> Option<Option<Packing>> {
        let OperandKind::Quantized(scheme) = self.kind else {
            return Some(None);
        };
        let along = Operand::dimension_of(&scheme, dimensions)?;
        let operand = Operand::of_scheme(&scheme, along, dimensions)?;
        let operand = match self.layout {
            OperandLayout::Tiles(_, tile, words, strips) => {
                // Tile order states a matrix `[k, rows]`: an `[m, k]` input states its tile
                // swapped, as it states its metadata.
                let (tile, across) = match dimensions {
                    [K, across] => (tile, across),
                    [across, _] => ((tile.1, tile.0), across),
                };
                let order = strips.laid(TileOrder::laid(&scheme, words, tile)?)?;
                let word = ElemType::UInt(UIntKind::U32);
                operand.in_tile_order(order, across, hw.load_values(word))?
            }
            OperandLayout::Along(_) | OperandLayout::Strided => operand,
        };
        Some(Some(Packing { operand, scheme }))
    }

    /// How this key lays `input`, a float input: in rows where it is contiguous along the second
    /// of the dimensions the product names it by, in columns along the first, strided along
    /// neither. `None` for one stored in tiles, which no dense layout states.
    pub fn float_layout(self, input: MatmulOperand) -> Option<MatmulLayout> {
        let [first, second] = input.dims().map(|dim| MatmulDimensions::MNK.of(dim));
        match self.layout {
            OperandLayout::Along(dimension) if dimension == second => Some(MatmulLayout::Row),
            OperandLayout::Along(dimension) if dimension == first => Some(MatmulLayout::Col),
            OperandLayout::Along(_) | OperandLayout::Tiles(..) => None,
            OperandLayout::Strided => Some(MatmulLayout::Strided),
        }
    }

    /// A float input contiguous along `dimension`, strided where it names none, batched as
    /// `batching` says.
    pub fn float(dimension: Option<Dimension>, batching: MatmulBatching) -> Self {
        Self {
            kind: OperandKind::Float,
            layout: OperandLayout::along(dimension),
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
    /// The block hangs off an output dimension, and the grid pays for the share of
    /// every cube that falls outside the problem.
    Masked,
    /// The block tiles both output dimensions: every cube it distributes is full.
    Tiled,
}

impl MatmulShape {
    /// Output elements the product covers — the work a grid is cut from.
    pub fn work(&self) -> usize {
        self.m * self.n * self.batch
    }

    /// Cubes a grid of `block` holds: whole blocks along each output dimension,
    /// partial ones counted, and the batch riding on top.
    pub fn cubes(&self, block: CubeBlock) -> usize {
        self.m.div_ceil(block.m.max(1)) * self.n.div_ceil(block.n.max(1)) * self.batch
    }

    /// Output elements that grid **covers**, masking included — what the
    /// device pays for, against [`work`](Self::work), which is what it gets.
    pub fn covered(&self, block: CubeBlock) -> usize {
        self.cubes(block) * block.m.max(1) * block.n.max(1)
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
/// input contiguous along the dimension the key names. A col input, its buffer the view's
/// transpose, is read where it lies ([`MatmulTensors`](crate::kernels::MatmulTensors)), so its
/// widths are its own; an input the key names no dimension for, a strided view, is aligned to one
/// element.
impl From<&MatmulShape> for MatmulRoutineProblem {
    fn from(shape: &MatmulShape) -> Self {
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
            ..Self::dense(shape.m, shape.n, shape.k, shape.elem)
                .laid(
                    MatmulOperand::Lhs,
                    MatmulLayout::along(shape.lhs.layout.dimension(), MatmulOperand::Lhs),
                )
                .laid(
                    MatmulOperand::Rhs,
                    MatmulLayout::along(shape.rhs.layout.dimension(), MatmulOperand::Rhs),
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
    // rides the leading dimensions, not how it is spread.
    let MatmulProblem {
        m, k, n, batches, ..
    } = MatmulProblem::new((&lhs_shape, lhs_kind), (&rhs_shape, rhs_kind));
    // An operand keys by the dimension it is contiguous along, the tile it is stored in and the
    // order the tile's words lie in, so a winner elected on one layout is never served to another,
    // which its kernel may decline.
    let operand = |meta: &Metadata, kind: OperandKind, operand: MatmulOperand| {
        let shape = logical(meta);
        let order = match kind {
            OperandKind::Quantized(scheme) => TileOrder::of_matrix(&scheme, meta),
            _ => None,
        };
        let words = order.map_or(TileWords::RowByRow, TileOrder::words);
        let strips = order.map_or(TileStrips::BackToBack, TileStrips::of);
        OperandKey {
            kind,
            layout: match (contiguous(meta, operand.dimensions()), layout::tile(meta)) {
                (None, _) => OperandLayout::Strided,
                (Some(dimension), (1, 1)) => OperandLayout::Along(dimension),
                (Some(dimension), tile) => OperandLayout::Tiles(dimension, tile, words, strips),
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

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// A free dimension keys by the power of two above it, whatever the autotune
    /// level: the priority and the plan-time proof read the key's `m`, and a
    /// bucket represented by 87 rows would derive blocks nothing at 64 rows
    /// runs.
    #[test]
    fn a_free_dimension_keys_by_the_power_of_two_above_it() {
        assert_eq!(MatmulTuneKey::anchored(1, 8192), 1);
        assert_eq!(MatmulTuneKey::anchored(64, 8192), 64);
        assert_eq!(MatmulTuneKey::anchored(65, 8192), 128);
        assert_eq!(MatmulTuneKey::anchored(100_000, 8192), 8192);
        assert_eq!(MatmulTuneKey::anchored(0, 64), 1);
    }

    /// A float input is laid as its key says: in rows along the second of its dimensions, in
    /// columns along the first, strided along neither, and stated by no dense layout in tiles.
    #[test]
    fn a_float_input_is_laid_as_its_key_says() {
        let laid = |layout, input| {
            OperandKey {
                kind: OperandKind::Float,
                layout,
                batching: MatmulBatching::Shared,
            }
            .float_layout(input)
        };
        let (lhs, rhs) = (MatmulOperand::Lhs, MatmulOperand::Rhs);
        assert_eq!(laid(OperandLayout::Along(K), lhs), Some(MatmulLayout::Row));
        assert_eq!(laid(OperandLayout::Along(M), lhs), Some(MatmulLayout::Col));
        assert_eq!(laid(OperandLayout::Along(N), rhs), Some(MatmulLayout::Row));
        assert_eq!(laid(OperandLayout::Along(K), rhs), Some(MatmulLayout::Col));
        assert_eq!(
            laid(OperandLayout::Strided, lhs),
            Some(MatmulLayout::Strided)
        );
        let tiles = OperandLayout::Tiles(K, (8, 8), TileWords::RowByRow, TileStrips::BackToBack);
        assert_eq!(laid(tiles, rhs), None);
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

    /// The keys of a decode projection against a float col rhs and against a q4 rhs
    /// packed along `k`, as they stand: a key is what a saved autotune is looked up by, so a change
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
            "MatmulTuneKey { elem: Float(F16), k: 256, n: 64, m: 4, batches: 1, lhs: OperandKey { kind: Float, layout: K, batching: Shared }, rhs: OperandKey { kind: Quantized(QuantScheme { value: Q4S, store: PackedU32(1), mode: Symmetric, blocks: [Some(BlockScale { size: BlockSize { extents: [Values(1), Values(32), Values(1), Values(1), Values(1)], rest: Values(1) }, dtype: F16 }), None] }), layout: K, batching: Shared } }"
        );
    }
}
