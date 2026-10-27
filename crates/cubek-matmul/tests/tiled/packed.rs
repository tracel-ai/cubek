//! Packed matrices as the tiled tests build them: block-scaled values a quantizer reads back
//! exactly, bound as the words and scales a kernel reads.

use super::indexed::upload;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Axis;
use cubecl::zspace::Shape;
use cubecl_common::e4m3;
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor, WordsLaid};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::stored::Operand;
use cubek_std::matmul_axes::K;
use cubek_test_utils::{HostData, HostDataType, InputDataType, TestInput};

/// The per-tensor scale a two-level format is built under.
const GLOBAL_SCALE: f32 = 2.;

/// Block scales with one or two mantissa bits, which the quantizer and a `ue4m3` byte hold exactly.
const MANTISSAS: [f32; 4] = [1., 1.5, 2., 3.];

/// The block-scaled formats a packed test matrix is built in.
#[derive(Clone, Copy, Debug)]
pub(super) enum PackedFormat {
    Q8,
    Q4,
    Nvfp4,
}

impl PackedFormat {
    pub(super) fn scheme(self) -> QuantScheme {
        let blocks_of_32 = |value| {
            QuantScheme::default()
                .with_value(value)
                .per_block([32], ScaleDtype::F32)
                .with_store(QuantStore::PackedU32(0))
        };
        match self {
            Self::Q8 => blocks_of_32(QuantValue::Q8S),
            Self::Q4 => blocks_of_32(QuantValue::Q4S),
            Self::Nvfp4 => nvfp4_scheme(),
        }
    }
}

/// `slabs` matrices of `rows` rows of `k` values, laid one after the other along the axis of
/// `operand` beside `K`, packed along `k` in `format`.
#[derive(Clone, Copy, Debug)]
pub(super) struct PackedSlabsSpec {
    pub format: PackedFormat,
    pub operand: MatmulOperand,
    pub slabs: usize,
    pub rows: usize,
    pub k: usize,
    pub seed: usize,
}

/// A packed matrix of slabs with the dequantized values it holds. Its words run along `k`, a row
/// each, as a checkpoint stores them whichever axis the rows are.
pub(super) struct PackedSlabs {
    spec: PackedSlabsSpec,
    tensor: PackedTensor,
    values: Vec<f32>,
}

impl PackedSlabs {
    pub(super) fn new(client: &Client, spec: PackedSlabsSpec) -> Self {
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
        let global = (scheme.num_levels() > 1).then(|| upload(client, &[1, 1], &[GLOBAL_SCALE]));
        let values = match global {
            Some(_) => exact.into_iter().map(|v| v * GLOBAL_SCALE).collect(),
            None => exact,
        };
        let across = across(spec.operand);
        let tensor = PackedTensor::new(
            PlainTensor::of_handle(quantized.handle),
            WordsLaid::InRows,
            PlainTensor::of_handle(scales),
            global,
            scheme,
            Operand::of_scheme(&scheme, K, [across, K]).expect("one block along k"),
            [(across, rows), (K, spec.k)],
        );
        let slabs = Self {
            spec,
            tensor,
            values,
        };
        slabs.assert_distinct(&scheme);
        slabs
    }

    /// The matrix whole, its slabs not yet read as such.
    pub(super) fn matrix(&self) -> PackedTensor {
        self.tensor.clone()
    }

    /// The matrix read as the stack of its slabs.
    pub(super) fn stack(&self) -> StoredTensor {
        let stack = self
            .matrix()
            .stacked(self.spec.slabs, across(self.spec.operand))
            .expect("whole slabs of whole blocks");
        StoredTensor::Packed(stack)
    }

    /// The dequantized value at `k` and row `row` of slab `slab`.
    pub(super) fn cell(&self, slab: usize, k: usize, row: usize) -> f32 {
        self.values[(slab * self.spec.rows + row) * self.spec.k + k]
    }

    /// Slabs, rows and the words of a block all hold different values, so a read of the wrong one
    /// cannot match.
    fn assert_distinct(&self, scheme: &QuantScheme) {
        let PackedSlabsSpec { rows, k, .. } = self.spec;
        let slabs: Vec<&[f32]> = self.values.chunks(rows * k).collect();
        for (i, a) in slabs.iter().enumerate() {
            for b in &slabs[i + 1..] {
                assert!(mostly_differ(a, b), "two slabs hold the same values");
            }
        }
        assert!(
            mostly_differ(&self.values[..k], &self.values[k..2 * k]),
            "two rows hold the same values"
        );
        let words: Vec<&[f32]> = self.values[..block_of(scheme)]
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
    /// Values in blocks, each holding its peak and codes of the scheme under its own scale, so the
    /// quantizer rounds nothing and the host's values are the dequantized ones.
    fn exact_values(&self, scheme: &QuantScheme) -> Vec<f32> {
        let block = block_of(scheme);
        let peak = scheme.value.range().1;
        let codes: Vec<f32> = match scheme.value {
            QuantValue::E2M1 => vec![0., 0.5, 1., 1.5, 2., 3., 4., 6.],
            _ => (0..=peak as usize).map(|code| code as f32).collect(),
        };
        let headroom = peak.log2().ceil() as i32 + 2;
        (0..self.slabs * self.rows * self.k)
            .map(|i| {
                let (block_at, within) = (i / block, i % block);
                let h = hashed(block_at, self.seed);
                let scale = MANTISSAS[h % 4] * 2f32.powi(-headroom - ((h / 4) % 3) as i32);
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

/// The values one of `scheme`'s block scales covers along a row.
fn block_of(scheme: &QuantScheme) -> usize {
    scheme
        .block_size()
        .expect("a block-scaled scheme")
        .to_dim_vec(1)[0] as usize
}

/// The axis of `operand` beside `K`, which a packed matrix's rows run along.
fn across(operand: MatmulOperand) -> Axis {
    operand
        .axes()
        .into_iter()
        .find(|&axis| axis != K)
        .expect("an operand has an axis beside K")
}

/// `scheme` as the reference quantizer takes it, which has no per-tensor level and no `ue4m3`:
/// the words it writes are the scheme's own.
fn block_level(scheme: &QuantScheme) -> QuantScheme {
    QuantScheme::default()
        .with_value(scheme.value)
        .per_block([block_of(scheme) as u8], ScaleDtype::F32)
        .with_store(scheme.store)
}

/// `scales`, `f32` block scales of `[rows, blocks]`, stored in the type `scheme` states.
fn stored_scales(
    client: &Client,
    scheme: &QuantScheme,
    scales: TensorHandle,
    (rows, blocks): (usize, usize),
) -> TensorHandle {
    let stored = scheme.scale_dtype();
    if stored == ScaleDtype::F32 {
        return scales;
    }
    assert_eq!(
        stored,
        ScaleDtype::UE4M3,
        "only `f32` and `ue4m3` scales are built here"
    );
    let host = HostData::from_tensor_handle(client, scales, HostDataType::F32);
    let bytes: Vec<u8> = (0..rows * blocks)
        .map(|i| e4m3::from_f32(host.get_f32(&[i / blocks, i % blocks])).to_bits())
        .collect();
    TensorHandle::new_contiguous(
        Shape::from(vec![rows, blocks]),
        client.create_from_slice(&bytes),
        ElemType::from_scale_dtype(stored),
    )
}
