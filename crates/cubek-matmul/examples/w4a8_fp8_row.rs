//! W4A8 with its scales a row, as a prompt runs a model's projections: an MXFP4 weight with its
//! scales factored by row and an activation quantized to FP8-row, both on the device by
//! cubek-quant's [`per_row`](cubek_quant::per_row) launches, then multiplied through the autotune
//! their inputs select. The weight's codes decode into `e4m3` as the fill stages them, the product contracts
//! on the device's plain `e4m3` MMA, and both rows' scales come out of the sum.
//!
//! ```sh
//! cargo run --release -p cubek-matmul --example w4a8_fp8_row --features cubecl/hip
//! ```
//!
//! Each shape is Qwen3.8 27B's projection at 512 and 2048 rows, `[out, in]`; `W4A8_M` times one
//! count of rows. Each line is the autotune's winner, its device time over copies of the weight
//! that outgrow the cache, and its worst error over sampled cells, relative to the largest cell,
//! against the product of the inputs the launches quantized, in `f64`.

use std::time::Duration;

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
use cubek_matmul::kernels::plan::{MatmulCall, MatmulCallInput};
use cubek_matmul::kernels::select::MatmulOpStrategy;
use cubek_matmul::kernels::select::problem::MatmulProblem;
use cubek_matmul::kernels::select::selector::measured_autotune;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, RowScaledTensor, StoredTensor};
use cubek_matmul::kernels::tune::MatmulAutotuneGroups;
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulRoutine,
};
use cubek_matmul::tiled::storage;
use cubek_microkernel::matmul::cube::{CubeStagedMicrokernelSettings, MatmulPlaneLayout};
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings;
use cubek_microkernel::matmul::{MatmulOperand, MatmulStep};
use cubek_quant::per_row::{quantize_fp8_per_row, quantize_mxfp4_row_factored};
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order;
use cubek_std::matmul_dimensions::{K, N};
use cubek_std::microkernel::StageRing;
use half::f16;

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);
const WORD: ElemType = ElemType::UInt(UIntKind::U32);

/// Values of `k` a block scale covers.
const BLOCK: usize = 32;

/// Bytes of weights a timing cycles through, past the R9700's 64 MiB cache.
const ROTATION: usize = 256 << 20;

/// Cells of the output checked against the host.
const SAMPLES: usize = 512;

/// A product's inputs on the device, as its launch reads them.
struct Inputs {
    lhs: StoredTensor,
    rhs: StoredTensor,
    out: TensorHandle,
}

/// Normal values, one in 256 sixteen times larger, as a model's activations and weights hold
/// them, each row its own spread.
fn normal(count: usize, row: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|at| {
            let spread = 2f64.powi((at / row) as i32 % 5 - 2);
            let (u, v) = (next().max(1e-12), next());
            let gaussian = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            let outlier = if next() < 1.0 / 256.0 { 16.0 } else { 1.0 };
            (gaussian * spread * outlier * 0.05) as f32
        })
        .collect()
}

fn upload(client: &Client, values: &[f32], shape: [usize; 2]) -> TensorHandle {
    TensorHandle::new_contiguous(
        Shape::new(shape),
        client.create_from_slice(f32::as_bytes(values)),
        F32,
    )
}

fn read<T: CubeElement>(client: &Client, handle: &TensorHandle) -> Vec<T> {
    T::from_bytes(&client.read_one_unchecked_tensor(handle.clone().into_copy_descriptor())).to_vec()
}

/// `[rows, k]` FP8-row values read back, as `f64`.
fn fp8_row_values(
    client: &Client,
    values: &TensorHandle,
    scales: &TensorHandle,
    k: usize,
) -> Vec<f64> {
    let scales = read::<f32>(client, scales);
    read::<u8>(client, values)
        .iter()
        .enumerate()
        .map(|(at, &code)| e4m3(code) * f64::from(scales[at / k]))
        .collect()
}

/// `[rows, k]` MXFP4 values with row-factored scales read back, as `f64`: each code under its
/// block's offset and its
/// row's scale.
fn row_factored_values(
    client: &Client,
    codes: &TensorHandle,
    offsets: &TensorHandle,
    row_scales: &TensorHandle,
    k: usize,
) -> Vec<f64> {
    const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let (codes, offsets) = (read::<u32>(client, codes), read::<u8>(client, offsets));
    let row_scales = read::<f32>(client, row_scales);
    (0..codes.len() * 8)
        .map(|at| {
            let code = (codes[at / 8] >> (4 * (at % 8))) & 0xF;
            let magnitude = E2M1[(code & 0x7) as usize];
            let value = if code & 0x8 != 0 {
                -magnitude
            } else {
                magnitude
            };
            let scale =
                2f64.powi(i32::from(offsets[at / 32]) - 127) * f64::from(row_scales[at / k]);
            value * scale
        })
        .collect()
}

fn e4m3(code: u8) -> f64 {
    let (exponent, mantissa) = (i32::from((code >> 3) & 0xF), f64::from(code & 0x7));
    let magnitude = match exponent {
        0 => mantissa / 8.0 * 2f64.powi(-6),
        _ => (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7),
    };
    if code & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// The inputs of an `m × n × k` product, quantized on the device, the weight stored as `layout`
/// says, and, where `host` asks, their values on the host.
fn inputs(
    client: &Client,
    hw: &Hardware,
    (m, n, k): (usize, usize, usize),
    copy: u64,
    layout: Layout,
    host: bool,
) -> (Inputs, Option<(Vec<f64>, Vec<f64>)>) {
    let activation = upload(client, &normal(m * k, k, 1), [m, k]);
    let fp8 = quantize_fp8_per_row(client, &activation, F32);
    let lhs_values = host.then(|| fp8_row_values(client, &fp8.values, &fp8.scales, k));
    let row_scales = TensorHandle::new_contiguous(Shape::new([m, 1]), fp8.scales.handle, F32);
    let lhs = RowScaledTensor::new(
        PlainTensor::of_handle(fp8.values),
        PlainTensor::of_handle(row_scales),
        QuantScheme::fp8_per_row(),
    )
    .expect("an FP8-row activation");

    let weight = upload(client, &normal(n * k, k, 2 + copy), [n, k]);
    let mxfp4 = quantize_mxfp4_row_factored(client, &weight, F32);
    let scales = mxfp4.scales;
    let rhs_values = host.then(|| {
        row_factored_values(
            client,
            &mxfp4.codes,
            &scales.block_scales,
            &scales.row_scales,
            k,
        )
    });
    let lossy = read::<u32>(client, &scales.lossy_blocks)[0];
    if host && lossy > 0 {
        println!("  {lossy} blocks of the weight round as they decode into e4m3");
    }
    let row_scales = PlainTensor::of_handle(TensorHandle::new_contiguous(
        Shape::new([n, 1]),
        scales.row_scales.handle,
        F32,
    ));
    let rhs = match layout {
        Layout::Tiles => in_tile_order(
            client,
            hw,
            &mxfp4.codes,
            &scales.block_scales,
            row_scales,
            (n, k),
        ),
        Layout::Rows => {
            let scheme = QuantScheme::mxfp4_row_factored();
            PackedTensor::new(
                PlainTensor::of_handle(mxfp4.codes),
                PlainTensor::of_handle(scales.block_scales),
                Some(row_scales),
                scheme,
                Operand::of_scheme(&scheme, K, [N, K]).expect("blocks along k"),
                [(N, n), (K, k)],
            )
        }
    };
    let out = TensorHandle::zeros(client, [m, n], F16);
    let inputs = Inputs {
        lhs: StoredTensor::RowScaled(lhs),
        rhs: StoredTensor::Packed(rhs),
        out,
    };
    (inputs, lhs_values.zip(rhs_values))
}

/// Where the weight's words and block scales lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Each row's words and scales along `k`, as a checkpoint stores them.
    Rows,
    /// The tile order a load stores the weight in on this device, its scales beside it.
    Tiles,
}

/// The weight, `[n, k]` codes and block scales in rows, restored in the tile order a load stores
/// it in on `hw` ([`MatmulStep::stored_on`], [`storage::tile_order_on`]): its codes as the kernel
/// names them, `[n, k / 32, 32]`, and its block scales in tiles of their own where the order keeps
/// them there, else in their rows, under its scale a row.
fn in_tile_order(
    client: &Client,
    hw: &Hardware,
    codes: &TensorHandle,
    block_scales: &TensorHandle,
    row_scales: PlainTensor,
    (n, k): (usize, usize),
) -> PackedTensor {
    let in_rows = QuantScheme::mxfp4_row_factored();
    // The step a load stores the weight in, under the instruction that contracts it against an
    // `e4m3` activation.
    let contracted = ElemType::Float(FloatKind::E4M3);
    let read = Operand::of_scheme(&in_rows, K, [N, K]).expect("one block along k");
    let values = cubek_std::value_element(in_rows.value).expect("float codes");
    let step = MatmulStep::stored_on(hw, MatmulOperand::Rhs, read, values, contracted)
        .expect("the device contracts the weight");
    let tile = (step.along(K), step.along(N));
    // The same bytes stated over `[k, n]`, its words down `k`: the statement tile order stores.
    let scheme = QuantScheme::default()
        .with_value(in_rows.value)
        .per_block([BLOCK as u8, 1], ScaleDtype::UE8M0)
        .per_row(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(1));
    let order =
        storage::tile_order_on(hw, &scheme, tile, (k, n)).expect("tile order stores the weight");
    let words_per_load = hw.load_values(WORD);
    let blocks = k / BLOCK;
    let bound = |handle: &cubecl::server::Handle, meta: Metadata| TensorBinding {
        handle: handle.clone().binding(),
        strides: meta.strides,
        shape: meta.shape,
        tiling: meta.tiling,
    };
    let values = storage::tile(
        client,
        bound(
            &codes.handle,
            Metadata::new([n, blocks, BLOCK], [k, BLOCK, 1]),
        ),
        &tile_order::values_dimensions(N),
        WORD,
        Some(Field::from(scheme.value)),
        order.values_storage(N, words_per_load),
    )
    .expect("the values tile");
    let scales = match order.scales_storage(N, order.rows_a_load(N, words_per_load)) {
        Some(laid) => {
            let tiled = storage::tile(
                client,
                bound(
                    &block_scales.handle,
                    Metadata::new([n, blocks], [blocks, 1]),
                ),
                &tile_order::scales_dimensions(N),
                WORD,
                Some(Field::of_scale(ScaleDtype::UE8M0)),
                laid,
            )
            .expect("the scales tile");
            PlainTensor::new(bound(&tiled.handle, *tiled.metadata), WORD)
        }
        None => PlainTensor::of_handle(block_scales.clone()),
    };
    let matrix = tile_order::matrix_values(&values.metadata);
    let sized = [(K, k), (N, n)];
    let operand = Operand::of_scheme(&scheme, K, [K, N]).expect("one block along k");
    let operand =
        PackedTensor::read_in_tiles(hw, sized, operand, &matrix).expect("read in its tiles");
    PackedTensor::new(
        PlainTensor::new(bound(&values.handle, matrix), WORD),
        scales,
        Some(row_scales),
        scheme,
        operand,
        sized,
    )
}

fn call(
    client: &Client,
    hw: &Hardware,
    inputs: &Inputs,
    (m, n, k): (usize, usize, usize),
) -> MatmulCall {
    MatmulCall {
        hw: hw.clone(),
        client: client.clone(),
        tune_id: CubeTuneId::new(client, &cubecl::test_device()),
        lhs: MatmulCallInput {
            tensor: inputs.lhs.clone(),
            meta: Metadata::new([m, k], [k, 1]),
            framework_dequantize: None,
        },
        rhs: MatmulCallInput {
            tensor: inputs.rhs.clone(),
            // The weight `[n, k]` as the product reads it, `[k, n]`.
            meta: Metadata::new([k, n], [1, k]),
            framework_dequantize: None,
        },
        out: F16,
        destination: Some(inputs.out.clone()),
    }
}

fn main() {
    let device = cubecl::test_device();
    let client = device.client();
    let hw = Hardware::from_client(&client);
    let rows = std::env::var("W4A8_M")
        .ok()
        .and_then(|m| m.parse().ok())
        .map_or(vec![512, 2048], |m| vec![m]);
    let shapes = [
        ("qkv", 8192, 5120),
        ("o", 5120, 6144),
        ("gate/up", 34816, 5120),
        ("down", 5120, 17408),
    ];
    let layout = match std::env::var("W4A8_LAYOUT").as_deref() {
        Ok("rows") => Layout::Rows,
        _ => Layout::Tiles,
    };
    println!("weight in {layout:?}");
    for m in rows {
        for (name, n, k) in shapes {
            let dims = (m, n, k);
            let copies = ROTATION.div_ceil(n * k / 2).clamp(2, 8);
            let (first, host) = inputs(&client, &hw, dims, 0, layout, true);
            let (lhs, rhs) = host.expect("the first copy's values");
            let mut data = vec![first];
            data.extend(
                (1..copies as u64).map(|copy| inputs(&client, &hw, dims, copy, layout, false).0),
            );

            let probe = call(&client, &hw, &data[0], dims);
            let problem = MatmulProblem::new(
                (&Shape::new([m, k]), probe.lhs.tensor.kind()),
                (&Shape::new([k, n]), probe.rhs.tensor.kind()),
            );
            let groups = MatmulAutotuneGroups::default();
            let autotune = measured_autotune(MatmulOpStrategy::Autotune, &problem, &hw)
                .expect("FP8-row against MXFP4 with row-factored scales selects one autotune");
            // The first launch races the autotune's candidates and settles its winner.
            autotune.launch(&probe, groups);
            let winner = autotune
                .fastest(&probe, groups)
                .map_or("(none settled)".to_string(), |(identity, _)| {
                    identity.to_string()
                });

            let got: Vec<f64> = read::<f16>(&client, &data[0].out)
                .iter()
                .map(|v| v.to_f64())
                .collect();
            let cells: Vec<usize> = (0..SAMPLES)
                .map(|i| (i * 2_654_435_761) % (m * n))
                .collect();
            let expected: Vec<f64> = cells
                .iter()
                .map(|&cell| {
                    let (row, column) = (cell / n, cell % n);
                    (0..k)
                        .map(|kk| lhs[row * k + kk] * rhs[column * k + kk])
                        .sum()
                })
                .collect();
            let largest = expected.iter().fold(0f64, |a, v| a.max(v.abs()));
            let error = cells
                .iter()
                .zip(&expected)
                .map(|(&cell, want)| (got[cell] - want).abs() / largest)
                .fold(0f64, f64::max);

            let calls: Vec<MatmulCall> = data.iter().map(|d| call(&client, &hw, d, dims)).collect();
            let tuned = time(&client, || {
                for c in &calls {
                    autotune.launch(c, groups);
                }
            }) / calls.len() as u32;
            let tflops = |time: Duration| 2.0 * (m * n * k) as f64 / time.as_secs_f64() / 1e12;
            // The boxes the autotune lists for this product, each launched alone.
            let boxes = eight_bit_boxes()
                .into_iter()
                .filter_map(|settings| {
                    let launch = |d: &Inputs| {
                        CubeStagedMatmulRoutine.launch_into(
                            &hw,
                            &client,
                            d.lhs.clone(),
                            d.rhs.clone(),
                            PlainTensor::of_handle(d.out.clone()),
                            &settings,
                        )
                    };
                    launch(&data[0]).ok()?;
                    let time = time(&client, || {
                        for d in &data {
                            launch(d).expect("launched once already");
                        }
                    }) / data.len() as u32;
                    Some((tflops(time), settings))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0));
            println!(
                "{name:<8} m={m:<5} {:>6.1} TFLOP/s {:>8.1} µs  err {error:.1e}  autotune: {winner}",
                tflops(tuned),
                tuned.as_secs_f64() * 1e6
            );
            if let Some((best, settings)) = boxes {
                println!(
                    "{:<16} {best:>6.1} TFLOP/s  best 8-bit box alone: {settings}",
                    ""
                );
            }
        }
    }
}

/// The units' boxes the packed autotune lists for an 8-bit float product on a device whose
/// manual MMA multiplies `e4m3`: four by four fragments on four by two planes, one step deep and
/// two, a second stage in flight.
fn eight_bit_boxes() -> Vec<CubeStagedMatmulRoutineSettings> {
    [(4, 4, 1), (4, 4, 2)]
        .map(|(along_m, along_n, depth)| {
            CubeStagedMatmulRoutineSettings::on_grid(
                CubeStagedMicrokernelSettings::boxed(
                    PlaneFragmentMicrokernelSettings::grid(along_m, along_n, depth),
                    MatmulPlaneLayout::new(4, 2),
                )
                .ringed(StageRing::DOUBLE),
            )
        })
        .to_vec()
}

/// The device time of `launch`, the best of a few rounds, each launching it several times.
fn time(client: &Client, launch: impl Fn() + Send + Sync) -> Duration {
    launch();
    block_on(client.sync()).expect("sync");
    let mut best = Duration::MAX;
    for _ in 0..3 {
        let ((), duration) = client
            .profile(
                || {
                    for _ in 0..8 {
                        launch();
                    }
                },
                "w4a8 fp8-row",
            )
            .expect("profile");
        let ticks = block_on(duration.resolve()).expect("device ticks");
        best = best.min(ticks.duration() / 8);
    }
    best
}
