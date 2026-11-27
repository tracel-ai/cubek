//! Packed matrices as the tiled tests build them: block-scaled values a quantizer reads back
//! exactly, bound as the words and scales a kernel reads.

use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::Scalar;
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::{
    BlockExtent, BlockScale, BlockSize, QuantScheme, QuantStore, QuantValue, ScaleDtype,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Dimension;
use cubecl::tile::kind::Field;
use cubecl::tile::layout::Stagger;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubecl_common::{e4m3, ue8m0};
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_matmul::tiled::storage;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order::{self, TileOrder, TileWords};
use cubek_std::matmul_dimensions::{K, N};
use cubek_test_utils::{HostData, HostDataType, InputDataType, TestInput};

use crate::tiled::upload;

/// The scale a format with one for the whole matrix is built under.
const GLOBAL_SCALE: f32 = 2.;

/// The scales of a format with one for each row, dealt over the rows in turn.
const ROW_SCALES: [f32; 5] = [2., 0.25, 1., 4., 0.5];

/// Block scales with one or two mantissa bits, which the quantizer and a `ue4m3` byte hold exactly;
/// a `ue8m0` byte holds only the first.
const MANTISSAS: [f32; 4] = [1., 1.5, 2., 3.];

/// The block-scaled formats a packed test matrix is built in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackedFormat {
    Q8,
    Q4,
    Nvfp4,
    Mxfp4,
    /// MXFP4 under one more scale a row, its block scales at most one.
    Mxfp4RowFactored,
    Mxfp8,
}

impl PackedFormat {
    pub(crate) fn scheme(self) -> QuantScheme {
        let blocks_of_32 = |value| {
            QuantScheme::default()
                .with_value(value)
                .with_levels([BlockScale::per_block([32], ScaleDtype::F32)])
                .with_store(QuantStore::PackedU32(0))
        };
        match self {
            Self::Q8 => blocks_of_32(QuantValue::Q8S),
            Self::Q4 => blocks_of_32(QuantValue::Q4S),
            Self::Nvfp4 => nvfp4_scheme(),
            Self::Mxfp4 => QuantScheme::default()
                .with_value(QuantValue::E2M1)
                .with_levels([BlockScale::per_block([32], ScaleDtype::UE8M0)])
                .with_store(QuantStore::PackedU32(0)),
            Self::Mxfp4RowFactored => QuantScheme::mxfp4_row_factored(),
            Self::Mxfp8 => QuantScheme::default()
                .with_value(QuantValue::E4M3)
                .with_levels([BlockScale::per_block([32], ScaleDtype::UE8M0)])
                .with_store(QuantStore::PackedU32(0)),
        }
    }
}

/// How a test matrix's words lie: in rows, a row's words one run, or in tile order, in tiles of
/// `(k, rows)` whose words lie as the [`TileWords`] says.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PackedWords {
    InRows,
    InTiles(TileWords, (usize, usize)),
}

/// `slabs` matrices of `rows` rows of `k` values, laid one after the other along the dimension of
/// `operand` beside `K`, packed along `k` in `format`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedSlabsSpec {
    pub format: PackedFormat,
    pub operand: MatmulOperand,
    pub slabs: usize,
    pub rows: usize,
    pub k: usize,
    pub seed: usize,
}

/// A packed matrix of slabs with the dequantized values it holds. Its words run along `k`, a row
/// each, as a checkpoint stores them whichever dimension the rows are.
pub(crate) struct PackedSlabs {
    spec: PackedSlabsSpec,
    tensor: PackedTensor,
    host: HostSlabs,
}

/// The dequantized values of a packed matrix of slabs, on the host: what a routine reading it
/// multiplies.
#[derive(Debug, Clone)]
pub(crate) struct HostSlabs {
    rows: usize,
    k: usize,
    values: Vec<f32>,
}

impl HostSlabs {
    /// The value at `k` and row `row` of slab `slab`.
    pub(crate) fn cell(&self, slab: usize, k: usize, row: usize) -> f32 {
        self.values[(slab * self.rows + row) * self.k + k]
    }
}

impl PackedSlabs {
    pub(crate) fn new(client: &Client, spec: PackedSlabsSpec) -> Self {
        let scheme = spec.format.scheme();
        let rows = spec.slabs * spec.rows;
        let exact = spec.exact_values(&scheme);
        let quantized = TestInput::builder(client.clone(), Shape::from(vec![rows, spec.k]))
            .dtype(InputDataType::Quantized(block_level(&scheme)))
            .custom(exact.clone())
            .generate_test_tensor();
        let scales = stored_scales(
            client,
            &scheme,
            quantized.quantization.expect("a quantized tensor").scale,
            (rows, spec.k / block_of(&scheme)),
        );
        let outer_scales =
            outer_scales(&scheme, rows).map(|scales| upload(client, &[scales.len(), 1], &scales));
        let across = across(spec.operand);
        let tensor = PackedTensor::new(
            PlainTensor::of_handle(quantized.handle),
            PlainTensor::of_handle(scales),
            outer_scales,
            scheme,
            K,
            None,
            [(across, rows), (K, spec.k)],
        );
        let slabs = Self {
            spec,
            tensor,
            host: spec.dequantized(),
        };
        slabs.assert_distinct(&scheme);
        slabs
    }

    /// The matrix whole, its slabs not yet read as such.
    pub(crate) fn matrix(&self) -> PackedTensor {
        self.tensor.clone()
    }

    /// The matrix whole, its words lying as `words` says.
    pub(crate) fn stored(&self, client: &Client, words: PackedWords) -> PackedTensor {
        match words {
            PackedWords::InRows => self.matrix(),
            PackedWords::InTiles(words, tile) => self.in_tile_order(client, words, tile),
        }
    }

    /// The matrix read as the stack of its slabs.
    pub(crate) fn stack(&self) -> StoredTensor {
        let stack = self
            .matrix()
            .stacked(self.spec.slabs, across(self.spec.operand))
            .expect("whole slabs of whole blocks");
        StoredTensor::Packed(stack)
    }

    /// The matrix whole, an NVFP4 `[k, rows]` weight, its values and block scales laid in tile
    /// order as a writer lays one for this device's readers, its tiles `tile` and their words as
    /// `words` asks, its strips staggered where that pads them: the bytes a model's routed
    /// experts are stored in. Its slabs are not yet read as such.
    ///
    /// # Panics
    ///
    /// On a format other than NVFP4, or an operand other than the rhs.
    pub(crate) fn in_tile_order(
        &self,
        client: &Client,
        words: TileWords,
        tile: (usize, usize),
    ) -> PackedTensor {
        assert!(
            matches!(self.spec.format, PackedFormat::Nvfp4)
                && self.spec.operand == MatmulOperand::Rhs,
            "tile order is laid for an NVFP4 rhs here"
        );
        let scheme = tile_order::nvfp4_scheme();
        let hw = Hardware::from_client(client);
        let (k, n) = (self.spec.k, self.spec.slabs * self.spec.rows);
        let byte_bits = u8::BITS as usize;
        let transaction =
            Stagger::new(hw.transaction_bytes() * byte_bits, byte_bits).expect("bytes divide it");
        let order = TileOrder::laid(&scheme, words, tile)
            .expect("NVFP4 is tile order")
            .staggered(transaction)
            .for_matrix(k, n);
        let word = ElemType::UInt(UIntKind::U32);
        let words_per_load = hw.load_values(word);
        let blocks = k / order.block();
        let relayout = |source: TensorBinding, dimensions: &[Dimension], field, storage| {
            storage::tile(client, source, dimensions, word, Some(field), storage)
                .unwrap_or_else(|err| panic!("the stack lays out in tile order: {err:?}"))
        };
        // The values as the kernel names them, `[n, k/block, block]`, counted in values.
        let packed = self.tensor.words();
        let row_words = packed.binding.strides[0];
        let values = packed.bound_as(Metadata::new(
            [n, blocks, order.block()],
            [row_words * scheme.num_quants(), order.block(), 1],
        ));
        let values = relayout(
            values,
            &tile_order::values_dimensions(N),
            Field::from(scheme.value),
            order.values_storage(N, words_per_load),
        );
        // Row by row the scales stay the rows they are; block by block they are laid in tiles of
        // their own beside the values.
        let scales = self
            .tensor
            .scales()
            .next()
            .expect("a block-scaled matrix")
            .clone();
        let scales = match order.scales_storage(N, order.rows_a_load(N, words_per_load)) {
            None => scales,
            Some(storage) => PlainTensor::of_handle(relayout(
                scales.bound_as(Metadata::new([n, blocks], [blocks, 1])),
                &tile_order::scales_dimensions(N),
                Field::of_scale(block_scales(&scheme).dtype),
                storage,
            )),
        };
        let words = PlainTensor::of_handle(values.clone())
            .viewed(tile_order::matrix_values(&values.metadata));
        let sized = [(K, k), (N, n)];
        let operand = Operand::of_scheme(&scheme, K, [K, N]).expect("NVFP4 blocks k");
        let tiles = PackedTensor::read_in_tiles(&hw, sized, operand, &words.meta())
            .expect("a matrix laid in tile order reads in its tiles");
        PackedTensor::new(
            words,
            scales,
            self.tensor.scales().nth(1).cloned(),
            scheme,
            K,
            Some(tiles),
            sized,
        )
    }

    /// The dequantized value at `k` and row `row` of slab `slab`.
    pub(crate) fn cell(&self, slab: usize, k: usize, row: usize) -> f32 {
        self.host.cell(slab, k, row)
    }

    /// Slabs, rows and the words of a block all hold different values, so a read of the wrong one
    /// cannot match.
    fn assert_distinct(&self, scheme: &QuantScheme) {
        let PackedSlabsSpec { rows, k, .. } = self.spec;
        let slabs: Vec<&[f32]> = self.host.values.chunks(rows * k).collect();
        for (i, a) in slabs.iter().enumerate() {
            for b in &slabs[i + 1..] {
                assert!(mostly_differ(a, b), "two slabs hold the same values");
            }
        }
        assert!(
            mostly_differ(&self.host.values[..k], &self.host.values[k..2 * k]),
            "two rows hold the same values"
        );
        let words: Vec<&[f32]> = self.host.values[..block_of(scheme)]
            .chunks(scheme.num_quants())
            .collect();
        for (i, a) in words.iter().enumerate() {
            for b in &words[i + 1..] {
                assert_ne!(a, b, "two words of a block hold the same values");
            }
        }
    }
}

impl PackedSlabsSpec {
    /// The values the matrix holds, dequantized, computed on the host alone: the exact values,
    /// under the outer scales where the format has some.
    pub(crate) fn dequantized(&self) -> HostSlabs {
        let scheme = self.format.scheme();
        let exact = self.exact_values(&scheme);
        let values = match outer_scales(&scheme, self.slabs * self.rows) {
            Some(scales) => exact
                .chunks(self.k)
                .enumerate()
                .flat_map(|(row, values)| {
                    let scale = scales[row % scales.len()];
                    values.iter().map(move |value| value * scale)
                })
                .collect(),
            None => exact,
        };
        HostSlabs {
            rows: self.rows,
            k: self.k,
            values,
        }
    }

    /// Values in blocks, each holding its peak and codes of the scheme under its own scale, so the
    /// quantizer rounds nothing and the host's values are the dequantized ones.
    fn exact_values(&self, scheme: &QuantScheme) -> Vec<f32> {
        let block = block_of(scheme);
        let peak = scheme.value.range().1;
        let codes: Vec<f32> = match scheme.value {
            QuantValue::E2M1 => vec![0., 0.5, 1., 1.5, 2., 3., 4., 6.],
            QuantValue::E4M3 => e4m3_codes(),
            _ => (0..=peak as usize).map(|code| code as f32).collect(),
        };
        let headroom = peak.log2().ceil() as i32 + 2;
        (0..self.slabs * self.rows * self.k)
            .map(|i| {
                let (block_at, within) = (i / block, i % block);
                let h = hashed(block_at, self.seed);
                let mantissa = match block_scales(scheme).dtype {
                    ScaleDtype::UE8M0 => MANTISSAS[0],
                    _ => MANTISSAS[h % 4],
                };
                let scale = mantissa * 2f32.powi(-headroom - ((h / 4) % 3) as i32);
                let pick = hashed(i, self.seed) % (2 * codes.len());
                let code = match within {
                    0 => peak,
                    _ if pick < codes.len() => codes[pick],
                    _ => -codes[pick - codes.len()],
                };
                code * scale
            })
            .collect()
    }
}

/// Every finite non-negative value `e4m3` holds: its 127 codes below the NaN, so a block of them
/// quantizes to itself. The integers up to its peak do not: 17 already rounds.
fn e4m3_codes() -> Vec<f32> {
    (0..0x7f_u8)
        .map(|code| {
            let (exponent, mantissa) = (i32::from(code >> 3), f32::from(code & 0b111));
            match exponent {
                0 => mantissa / 8. * 2f32.powi(-6),
                _ => (1. + mantissa / 8.) * 2f32.powi(exponent - 7),
            }
        })
        .collect()
}

/// A well-mixed number from `i` and `seed`, so no pattern of the indices repeats at a slab, a row
/// or a word.
fn hashed(i: usize, seed: usize) -> usize {
    let mut x = (i as u64) ^ ((seed as u64) << 32);
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (x ^ (x >> 32)) as usize
}

/// Whether more than a quarter of the cells of `a` and `b` differ.
fn mostly_differ(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).filter(|(a, b)| a != b).count() * 4 > a.len()
}

/// `scheme`'s innermost level: the block scales laid beside its words.
pub(crate) fn block_scales(scheme: &QuantScheme) -> BlockScale {
    scheme
        .levels()
        .next()
        .expect("a packed test scheme should state its block scales")
}

/// The values one of `scheme`'s block scales covers along a row.
pub(crate) fn block_of(scheme: &QuantScheme) -> usize {
    match block_scales(scheme).size.extents[0] {
        BlockExtent::Values(values) => values.get() as usize,
        BlockExtent::WholeAxis => panic!("a block-scaled scheme"),
    }
}

/// The scales of `scheme`'s outer level for a matrix of `rows` rows, where it has one: one over all
/// of it, or one for each row.
fn outer_scales(scheme: &QuantScheme, rows: usize) -> Option<Vec<f32>> {
    let outer = scheme.levels().nth(1)?;
    Some(match outer.size {
        BlockSize::TENSOR => vec![GLOBAL_SCALE],
        _ => (0..rows)
            .map(|row| ROW_SCALES[row % ROW_SCALES.len()])
            .collect(),
    })
}

/// The dimension of `operand` beside `K`, which a packed matrix's rows run along.
fn across(operand: MatmulOperand) -> Dimension {
    operand
        .dimensions()
        .into_iter()
        .find(|&dimension| dimension != K)
        .expect("an operand has a dimension beside K")
}

/// `scheme` as the reference quantizer takes it, which has no per-tensor level and no `ue4m3`:
/// the words it writes are the scheme's own.
fn block_level(scheme: &QuantScheme) -> QuantScheme {
    QuantScheme::default()
        .with_value(scheme.value)
        .with_levels([BlockScale::per_block(
            [block_of(scheme) as u8],
            ScaleDtype::F32,
        )])
        .with_store(scheme.store)
}

/// `scales`, `f32` block scales of `[rows, blocks]`, stored in the type `scheme` states.
fn stored_scales(
    client: &Client,
    scheme: &QuantScheme,
    scales: TensorHandle,
    (rows, blocks): (usize, usize),
) -> TensorHandle {
    let stored = block_scales(scheme).dtype;
    if stored == ScaleDtype::F32 {
        return scales;
    }
    let host = HostData::from_tensor_handle(client, scales, HostDataType::F32);
    let byte = |scale: f32| match stored {
        ScaleDtype::UE4M3 => e4m3::from_f32(scale).to_bits(),
        ScaleDtype::UE8M0 => ue8m0::from_f32(scale).to_bits(),
        other => panic!("only `f32`, `ue4m3` and `ue8m0` scales are built here, not {other:?}"),
    };
    let bytes: Vec<u8> = (0..rows * blocks)
        .map(|i| byte(host.get_f32(&[i / blocks, i % blocks])))
        .collect();
    TensorHandle::new_contiguous(
        Shape::from(vec![rows, blocks]),
        client.create_from_slice(&bytes),
        ElemType::from_scale_dtype(stored),
    )
}
