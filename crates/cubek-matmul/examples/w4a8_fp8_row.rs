//! W4A8 with its scales a row at a model's prefill: an activation quantized to FP8-row and a
//! weight to MXFP4 with row-factored scales, both on the device, the weight stored in the tile
//! order a load reads it in, then multiplied through the plan the selection picks, which races
//! its candidates on the first launch. Each line is a Qwen3.8 27B projection `[out, in]` at a
//! prompt's rows: its TFLOP/s over copies of the weight that outgrow the cache, and its worst
//! error over sampled cells against the product of the quantized inputs in `f64`.
//!
//! ```sh
//! cargo run --release -p cubek-matmul --example w4a8_fp8_row --features cubecl/hip
//! ```

use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::future::block_on;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantScheme, QuantStore, ScaleDtype};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Field;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubecl_common::{e2m1, e4m3};
use cubek_matmul::kernels::plan::{MatmulCall, MatmulCallInput};
use cubek_matmul::kernels::select::MatmulOpStrategy;
use cubek_matmul::kernels::select::problem::MatmulProblem;
use cubek_matmul::kernels::select::selector::select;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, RowScaledTensor, StoredTensor};
use cubek_matmul::kernels::tune::MatmulAutotuneGroups;
use cubek_matmul::tiled::storage;
use cubek_microkernel::matmul::{MatmulOperand, MatmulStep};
use cubek_quant::per_row::{MXFP4_BLOCK, quantize_fp8_per_row, quantize_mxfp4_row_factored};
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order;
use cubek_std::matmul_dimensions::{K, N};
use half::f16;

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);
const WORD: ElemType = ElemType::UInt(UIntKind::U32);
/// Bytes of weights the timing cycles through, past the R9700's 64 MiB cache.
const ROTATION: usize = 256 << 20;
/// Launches a timing averages.
const LAUNCHES: usize = 32;
/// Cells of the output checked against the host.
const SAMPLES: usize = 256;

fn main() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let shapes = [
        ("qkv", 8192, 5120),
        ("o", 5120, 6144),
        ("gate/up", 34816, 5120),
        ("down", 5120, 17408),
    ];
    for m in [512, 2048] {
        for (name, n, k) in shapes {
            let lhs = activation(&client, m, k);
            let copies = ROTATION.div_ceil(n * k / 2).clamp(2, 8);
            let weights: Vec<_> = (0..copies)
                .map(|copy| weight(&client, &hw, n, k, copy))
                .collect();
            let calls: Vec<MatmulCall> = weights
                .iter()
                .map(|(rhs, _)| call(&client, &hw, lhs.0.clone(), rhs.clone(), (m, n, k)))
                .collect();
            let problem = MatmulProblem::new(
                (&Shape::new([m, k]), lhs.0.kind()),
                (&Shape::new([k, n]), weights[0].0.kind()),
            );
            let plan = select(
                MatmulOpStrategy::Autotune,
                &problem,
                &hw,
                MatmulAutotuneGroups::default(),
            )
            .expect("FP8-row against MXFP4 with row-factored scales has a plan");
            // The first launch races the candidates.
            let out = plan.launch(&calls[0]).expect("a plan serves the product");
            let error = worst_error(&client, &out, &lhs.1, &weights[0].1, (m, n, k));
            block_on(client.sync()).expect("sync");
            let start = std::time::Instant::now();
            for i in 0..LAUNCHES {
                plan.launch(&calls[i % copies])
                    .expect("served once already");
            }
            block_on(client.sync()).expect("sync");
            let seconds = start.elapsed().as_secs_f64() / LAUNCHES as f64;
            let tflops = 2.0 * (m * n * k) as f64 / seconds / 1e12;
            println!("{name:<8} m={m:<5} {tflops:>6.1} TFLOP/s  err {error:.1e}");
        }
    }
}

/// Normal values, one in 256 sixteen times larger, as a model's activations and weights hold
/// them.
fn normal(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let (u, v) = (uniform().max(1e-12), uniform());
            let gaussian = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            let outlier = if uniform() < 1.0 / 256.0 { 16.0 } else { 1.0 };
            (gaussian * outlier * 0.05) as f32
        })
        .collect()
}

fn upload(client: &Client, values: &[f32], shape: [usize; 2]) -> TensorHandle {
    let handle = client.create_from_slice(f32::as_bytes(values));
    TensorHandle::new_contiguous(Shape::new(shape), handle, F32)
}

fn read<T: CubeElement>(client: &Client, handle: &TensorHandle) -> Vec<T> {
    let bytes = client.read_one_unchecked_tensor(handle.clone().into_copy_descriptor());
    T::from_bytes(&bytes).to_vec()
}

/// An `[m, k]` activation quantized to FP8-row, and its values on the host.
fn activation(client: &Client, m: usize, k: usize) -> (StoredTensor, Vec<f64>) {
    let fp8 = quantize_fp8_per_row(client, &upload(client, &normal(m * k, 1), [m, k]), F32);
    let scales = read::<f32>(client, &fp8.scales);
    let host = read::<u8>(client, &fp8.values)
        .iter()
        .enumerate()
        .map(|(at, &code)| f64::from(e4m3::from_bits(code).to_f32() * scales[at / k]))
        .collect();
    let row_scales = TensorHandle::new_contiguous(Shape::new([m, 1]), fp8.scales.handle, F32);
    let lhs = RowScaledTensor::new(
        PlainTensor::of_handle(fp8.values),
        PlainTensor::of_handle(row_scales),
        QuantScheme::fp8_per_row(),
    )
    .expect("an FP8-row activation");
    (StoredTensor::RowScaled(lhs), host)
}

/// An `[n, k]` weight quantized to MXFP4 with row-factored scales, stored in the tile order a
/// load reads it in on `hw` ([`MatmulStep::stored_on`], [`storage::tile_order_on`]), and its
/// values on the host.
fn weight(
    client: &Client,
    hw: &Hardware,
    n: usize,
    k: usize,
    copy: usize,
) -> (StoredTensor, Vec<f64>) {
    let weight = upload(client, &normal(n * k, 2 + copy as u64), [n, k]);
    let mxfp4 = quantize_mxfp4_row_factored(client, &weight, F32);
    let (codes, scales) = (mxfp4.codes, mxfp4.scales);
    let host = host_weight(client, &codes, &scales.block_scales, &scales.row_scales, k);
    // The scheme over `[k, n]`, its words down `k`: the statement tile order stores.
    let scheme = QuantScheme::mxfp4_row_factored();
    let along_k = QuantScheme::default()
        .with_value(scheme.value)
        .per_block([MXFP4_BLOCK as u8, 1], ScaleDtype::UE8M0)
        .per_row(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(1));
    let read = Operand::of_scheme(&scheme, K, [N, K]).expect("blocks along k");
    let element = cubek_std::value_element(scheme.value).expect("float codes");
    let contracted = ElemType::Float(FloatKind::E4M3);
    let step = MatmulStep::stored_on(hw, MatmulOperand::Rhs, read, element, contracted)
        .expect("the device contracts the weight");
    let order = storage::tile_order_on(hw, &along_k, (step.along(K), step.along(N)), (k, n))
        .expect("tile order stores the weight");
    let words_per_load = hw.load_values(WORD);
    let blocks = k / MXFP4_BLOCK;
    let bound = |handle: &cubecl::server::Handle, meta: Metadata| TensorBinding {
        handle: handle.clone().binding(),
        strides: meta.strides,
        shape: meta.shape,
        tiling: meta.tiling,
    };
    let tiled = |handle: &cubecl::server::Handle, meta, dimensions: &[_], field, laid| {
        storage::tile(
            client,
            bound(handle, meta),
            dimensions,
            WORD,
            Some(field),
            laid,
        )
        .expect("the weight tiles")
    };
    let values = tiled(
        &codes.handle,
        Metadata::new([n, blocks, MXFP4_BLOCK], [k, MXFP4_BLOCK, 1]),
        &tile_order::values_dimensions(N),
        Field::from(scheme.value),
        order.values_storage(N, words_per_load),
    );
    let block_scales = match order.scales_storage(N, order.rows_a_load(N, words_per_load)) {
        Some(laid) => {
            let scales = tiled(
                &scales.block_scales.handle,
                Metadata::new([n, blocks], [blocks, 1]),
                &tile_order::scales_dimensions(N),
                Field::of_scale(ScaleDtype::UE8M0),
                laid,
            );
            PlainTensor::new(bound(&scales.handle, *scales.metadata), WORD)
        }
        None => PlainTensor::of_handle(scales.block_scales),
    };
    let matrix = tile_order::matrix_values(&values.metadata);
    let sized = [(K, k), (N, n)];
    let operand = Operand::of_scheme(&along_k, K, [K, N]).expect("blocks along k");
    let operand =
        PackedTensor::read_in_tiles(hw, sized, operand, &matrix).expect("read in its tiles");
    let row_scales =
        TensorHandle::new_contiguous(Shape::new([n, 1]), scales.row_scales.handle, F32);
    let rhs = PackedTensor::new(
        PlainTensor::new(bound(&values.handle, matrix), WORD),
        block_scales,
        Some(PlainTensor::of_handle(row_scales)),
        along_k,
        operand,
        sized,
    );
    (StoredTensor::Packed(rhs), host)
}

/// `[n, k]` MXFP4 values with row-factored scales, each code under its block's offset and its
/// row's scale.
fn host_weight(
    client: &Client,
    codes: &TensorHandle,
    offsets: &TensorHandle,
    row_scales: &TensorHandle,
    k: usize,
) -> Vec<f64> {
    let (codes, offsets) = (read::<u32>(client, codes), read::<u8>(client, offsets));
    let row_scales = read::<f32>(client, row_scales);
    (0..codes.len() * 8)
        .map(|at| {
            let code = ((codes[at / 8] >> (4 * (at % 8))) & 0xF) as u8;
            let offset = 2f64.powi(i32::from(offsets[at / MXFP4_BLOCK]) - 127);
            f64::from(e2m1::from_bits(code).to_f32()) * offset * f64::from(row_scales[at / k])
        })
        .collect()
}

/// The call of `lhs · rhs`, `rhs` the weight `[n, k]` as the product reads it, `[k, n]`.
fn call(
    client: &Client,
    hw: &Hardware,
    lhs: StoredTensor,
    rhs: StoredTensor,
    (m, n, k): (usize, usize, usize),
) -> MatmulCall {
    let input = |tensor, meta| MatmulCallInput {
        tensor,
        meta,
        framework_dequantize: None,
    };
    MatmulCall {
        hw: hw.clone(),
        client: client.clone(),
        tune_id: CubeTuneId::new(client, &cubecl::test_device()),
        lhs: input(lhs, Metadata::new([m, k], [k, 1])),
        rhs: input(rhs, Metadata::new([k, n], [1, k])),
        out: F16,
        destination: None,
    }
}

/// The worst error of `out` over sampled cells against `lhs · rhs` in `f64`, relative to the
/// largest of them.
fn worst_error(
    client: &Client,
    out: &TensorHandle,
    lhs: &[f64],
    rhs: &[f64],
    (m, n, k): (usize, usize, usize),
) -> f64 {
    let got = read::<f16>(client, out);
    let cells = (0..SAMPLES).map(|i| (i * 2_654_435_761) % (m * n));
    let pairs: Vec<(f64, f64)> = cells
        .map(|cell| {
            let (row, column) = (cell / n, cell % n);
            let want = (0..k).map(|i| lhs[row * k + i] * rhs[column * k + i]).sum();
            (got[cell].to_f64(), want)
        })
        .collect();
    let largest = pairs.iter().fold(0f64, |a, (_, want)| a.max(want.abs()));
    pairs
        .iter()
        .fold(0f64, |a, (got, want)| a.max((got - want).abs()))
        / largest
}
