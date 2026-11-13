//! Packed matrices as the tiled tests build them: block-scaled values a quantizer reads back
//! exactly, bound as the words and scales a kernel reads.

use super::indexed::upload;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::Scalar;
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Axis;
use cubecl::tile::kind::Field;
use cubecl::tile::layout::Stagger;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubecl_common::{e4m3, ue8m0};
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_matmul::kernels::{CubeShuffleMatmulRoutine, MatmulRoutine, PlaneShuffleMatmulRoutine};
use cubek_matmul::tiled::storage;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order::{self, TileOrder, TileWords};
use cubek_std::matmul_axes::{K, N};
use cubek_test_utils::{HostData, HostDataType, InputDataType, TestInput};
use half::bf16;

/// The per-tensor scale a two-level format is built under.
const GLOBAL_SCALE: f32 = 2.;

/// Loose enough for a `bf16` output.
const BF16_TOLERANCE: f32 = 1e-2;

/// Block scales with one or two mantissa bits, which the quantizer and a `ue4m3` byte hold exactly;
/// a `ue8m0` byte holds only the first.
const MANTISSAS: [f32; 4] = [1., 1.5, 2., 3.];

/// The block-scaled formats a packed test matrix is built in.
#[derive(Clone, Copy, Debug)]
pub(super) enum PackedFormat {
    Q8,
    Q4,
    Nvfp4,
    Mxfp4,
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
            Self::Mxfp4 => QuantScheme::default()
                .with_value(QuantValue::E2M1)
                .per_block([32], ScaleDtype::UE8M0)
                .with_store(QuantStore::PackedU32(0)),
        }
    }
}

/// How a test matrix's words lie: in rows, a row's words one run, or in tile order, in tiles of
/// `(k, rows)` whose words lie as the [`TileWords`] says.
#[derive(Clone, Copy, Debug)]
pub(super) enum Laid {
    InRows,
    InTiles(TileWords, (usize, usize)),
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

    /// The matrix whole, its words lying as `laid` says.
    pub(super) fn laid(&self, client: &Client, laid: Laid) -> PackedTensor {
        match laid {
            Laid::InRows => self.matrix(),
            Laid::InTiles(words, tile) => self.in_tile_order(client, words, tile),
        }
    }

    /// The matrix read as the stack of its slabs.
    pub(super) fn stack(&self) -> StoredTensor {
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
    pub(super) fn in_tile_order(
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
        let transaction = Stagger::new(hw.transaction_bytes() * 8, 8).expect("bytes divide it");
        let order = TileOrder::laid(&scheme, words, tile)
            .expect("NVFP4 is tile order")
            .staggered(transaction)
            .for_matrix(k, n);
        let word = ElemType::UInt(UIntKind::U32);
        let words_per_load = hw.load_values(word);
        let blocks = k / order.block();
        let relayout = |source: TensorBinding, axes: &[Axis], field, storage| {
            storage::tile(client, source, axes, word, Some(field), storage)
                .unwrap_or_else(|err| panic!("the stack lays out in tile order: {err:?}"))
        };
        // The values as the kernel names them, `[n, k/block, block]`, counted in values.
        let packed = self.tensor.words();
        let values = packed.bound_as(Metadata::new(
            [n, blocks, order.block()],
            [
                packed.binding.strides[0] * scheme.num_quants(),
                order.block(),
                1,
            ],
        ));
        let values = relayout(
            values,
            &tile_order::values_axes(N),
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
                &tile_order::scales_axes(N),
                Field::of_scale(scheme.scale_dtype()),
                storage,
            )),
        };
        let words = PlainTensor::of_handle(values.clone())
            .viewed(tile_order::matrix_values(&values.metadata));
        let sized = [(K, k), (N, n)];
        let operand = Operand::of_scheme(&scheme, K, [K, N]).expect("NVFP4 blocks k");
        let operand = PackedTensor::read_in_tiles(&hw, sized, operand, &words.meta())
            .expect("a matrix laid in tile order reads in its tiles");
        PackedTensor::new(
            words,
            scales,
            self.tensor.scales().nth(1).cloned(),
            scheme,
            operand,
            sized,
        )
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
                let mantissa = match scheme.scale_dtype() {
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

/// A packed tensor counts the bytes its words and scales bind, not the buffer they share: a
/// framework stores the three in one, each bound over its own region of it. An autotune's
/// roofline reads these bytes, and counting the shared buffer once a part tripled them.
#[test]
fn a_packed_tensor_binds_its_parts_not_the_buffer_they_share() {
    let client = cubek_test_utils::client();
    let scheme = nvfp4_scheme();
    let (rows, k) = (16, 64);
    let word = ElemType::UInt(cubecl::ir::UIntKind::U32).size();
    let words = rows * k / scheme.num_quants();
    let scale_words = rows * k / block_of(&scheme) / word;
    let total = (words + scale_words + 1) * word;
    let buffer = client.empty(total);
    let part = |from: usize, words: usize, shape: [usize; 2], elem: ElemType| {
        let handle = buffer
            .clone()
            .offset_start(from as u64)
            .offset_end((total - from - words * word) as u64);
        PlainTensor::of_handle(TensorHandle::new_contiguous(
            Shape::from(shape.to_vec()),
            handle,
            elem,
        ))
    };
    let u32 = ElemType::UInt(cubecl::ir::UIntKind::U32);
    let packed = PackedTensor::new(
        part(0, words, [rows, words / rows], u32),
        part(words * word, scale_words, [rows, scale_words / rows], u32),
        Some(part((words + scale_words) * word, 1, [1, 1], u32)),
        scheme,
        Operand::of_scheme(&scheme, K, [N, K]).expect("one block along k"),
        [(N, rows), (K, k)],
    );

    assert_eq!(StoredTensor::Packed(packed).bytes(), total);
}

/// A packed matrix seen through a transposed view, as a framework hands a tied embedding to the
/// vocabulary head: `[n, k]` packed along `k`, bound as `[k, n]` under swapped strides, its words
/// running along the view's first axis. [`PackedTensor::of_quantized`] reads it as the matrix it
/// views, and the routines a decode step races answer its product. It refused the view once, its
/// words not row-major under it, and the framework dequantized the whole head on every step.
#[test]
fn a_transposed_packed_matrix_is_read_as_the_matrix_it_views() {
    let (m, n, k) = (4, 96, 512);
    let client = cubek_test_utils::client();
    let hw = Hardware::from_client(&client);
    let weight = PackedSlabs::new(
        &client,
        PackedSlabsSpec {
            format: PackedFormat::Nvfp4,
            operand: MatmulOperand::Rhs,
            slabs: 1,
            rows: n,
            k,
            seed: 3,
        },
    );
    let matrix = weight.matrix();
    // The view's scheme, as a framework's transpose restates it: blocks and words along its
    // first axis.
    let mut scheme = matrix.scheme();
    scheme.swap_block_dims(2, 0, 1);
    scheme.store = QuantStore::PackedU32(1);
    let mut values = matrix.words().transposed().binding;
    values.shape[0] = k;
    let mut scales = matrix.scales().cloned();
    let block_scales = scales.next().expect("block scales").transposed();
    let rhs =
        PackedTensor::of_quantized(&client, values, scheme, block_scales, scales.next(), [K, N])
            .expect("a transposed view of a packed matrix reads as stored");

    let lhs: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
        .collect();
    let expected: Vec<f32> = (0..m * n)
        .map(|at| {
            let (row, column) = (at / n, at % n);
            (0..k)
                .map(|kk| lhs[row * k + kk].to_f32() * weight.cell(0, kk, column))
                .sum()
        })
        .collect();
    let mut ran = Vec::new();
    let mut wrong = Vec::new();
    let mut race = |launch: &dyn Fn(StoredTensor, PlainTensor) -> Option<String>| {
        let (out, bound) = PlainTensor::empty_contiguous(
            &client,
            Shape::from(vec![m, n]),
            bf16::elem_type_native(),
        );
        let Some(settings) = launch(StoredTensor::Packed(rhs.clone()), bound) else {
            return;
        };
        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let off = expected
            .iter()
            .enumerate()
            .filter(|&(at, &want)| {
                (got.get_f32(&[at / n, at % n]) - want).abs() > BF16_TOLERANCE * (1. + want.abs())
            })
            .count();
        if off > 0 {
            wrong.push(format!("{settings}: {off} wrong"));
        }
        ran.push(settings);
    };
    macro_rules! raced {
        ($routine:expr, $settings:expr) => {
            for settings in $settings {
                race(&|rhs, out| {
                    $routine
                        .launch_into(
                            &hw,
                            &client,
                            StoredTensor::Plain(upload(&client, &[m, k], &lhs)),
                            rhs,
                            out,
                            &settings,
                        )
                        .ok()
                        .map(|_| settings.to_string())
                });
            }
        };
    }
    raced!(
        PlaneShuffleMatmulRoutine,
        PlaneShuffleMatmulRoutine.packed_settings()
    );
    raced!(
        CubeShuffleMatmulRoutine,
        CubeShuffleMatmulRoutine.packed_settings(&hw)
    );

    assert!(!ran.is_empty(), "no shuffle setting prepares a packed rhs");
    assert!(wrong.is_empty(), "{wrong:#?}");
}
