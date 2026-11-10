//! cubek's NVFP4 gemv as a decode step runs it: one `f16` row against a packed `[n, k]` weight,
//! Qwen3-8B's projections, every setting of the routines the few-rows autotune races against a
//! packed rhs, each checked against an `f64` reference.
//!
//! Every launch reads its operands cold (a rotation through 512 MB of copies), in device time, the
//! best of interleaved rounds, in GB/s of the weight's words and block scales.
//!
//! ```sh
//! cargo run --release -p cubek-matmul --example nvfp4_gemv --features cubecl/hip
//! ```
//!
//! `NVFP4_M` sets the rows (1), `NVFP4_ROUNDS` the rounds (3), `NVFP4_SHAPE` keeps one shape,
//! `NVFP4_ONLY` keeps the settings whose name contains it.

use std::time::Duration;

use cubecl::client::Client;
use cubecl::future::block_on;
use cubecl::ir::{ElemType, FloatKind, UIntKind};

use cubecl::quant::scheme::ScaleDtype;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_matmul::kernels::{CubeShuffleMatmulRoutine, MatmulRoutine, PlaneShuffleMatmulRoutine};
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_axes::{K, N};
use half::f16;

const BLOCK: usize = 16;
const CODES_PER_WORD: usize = 8;
const SCALES_PER_WORD: usize = 4;
const WORD: ElemType = ElemType::UInt(UIntKind::U32);
const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);
/// Twice the largest last-level cache in this class.
const ROTATION: usize = 512 << 20;

const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn ue4m3(byte: u8) -> f64 {
    let (exp, mant) = ((byte >> 3) & 0xF, byte & 0x7);
    match exp {
        0 => (mant as f64 / 8.0) * 2f64.powi(-6),
        _ => (1.0 + mant as f64 / 8.0) * 2f64.powi(exp as i32 - 7),
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The weight, the row it is multiplied by, and the product, on the host.
struct Host {
    words: Vec<u32>,
    scale_words: Vec<u32>,
    global: f32,
    x: Vec<f16>,
    expected: Vec<f64>,
}

impl Host {
    fn new(m: usize, n: usize, k: usize) -> Self {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let codes: Vec<u8> = (0..n * k).map(|_| (rng.next() % 16) as u8).collect();
        // Scales from 2^-3 to 2^4, never 0x7F, `ue4m3`'s NaN.
        let scales: Vec<u8> = (0..n * k / BLOCK)
            .map(|_| (0x20 + rng.next() % 0x40) as u8)
            .collect();
        let global = 0.371f32;
        // Eighths in [-1, 1]: exact in `f16`.
        let x: Vec<f16> = (0..m * k)
            .map(|_| f16::from_f32((rng.next() % 17) as f32 / 8.0 - 1.0))
            .collect();
        let words = codes
            .chunks(CODES_PER_WORD)
            .map(|w| {
                w.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &c)| acc | ((c as u32) << (4 * j)))
            })
            .collect();
        let scale_words = scales
            .chunks(SCALES_PER_WORD)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .collect();
        let blocks = k / BLOCK;
        let expected = (0..m * n)
            .map(|at| {
                let (x_row, row) = (at / n, at % n);
                (0..blocks)
                    .map(|b| {
                        let block: f64 = (0..BLOCK)
                            .map(|i| {
                                let at = b * BLOCK + i;
                                let code = codes[row * k + at];
                                let m = E2M1[(code & 7) as usize];
                                let v = if code & 8 != 0 { -m } else { m };
                                v * x[x_row * k + at].to_f64()
                            })
                            .sum();
                        block * ue4m3(scales[row * blocks + b])
                    })
                    .sum::<f64>()
                    * global as f64
            })
            .collect();
        Host {
            words,
            scale_words,
            global,
            x,
            expected,
        }
    }
}

/// One copy of the operands on the device.
struct Device {
    weight: PackedTensor,
    x: PlainTensor,
    out: TensorHandle,
}

fn tensor(client: &Client, bytes: &[u8], shape: [usize; 2], elem: ElemType) -> TensorHandle {
    TensorHandle::new_contiguous(
        Shape::from(shape.to_vec()),
        client.create_from_slice(bytes),
        elem,
    )
}

impl Device {
    fn new(client: &Client, host: &Host, m: usize, n: usize, k: usize) -> Self {
        let scheme = nvfp4_scheme();
        let words: Vec<u8> = host.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales: Vec<u8> = host
            .scale_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let weight = PackedTensor::new(
            PlainTensor::of_handle(tensor(client, &words, [n, k / CODES_PER_WORD], WORD)),
            // A byte a block, as a checkpoint stores them: four share a word in memory.
            PlainTensor::of_handle(tensor(
                client,
                &scales,
                [n, k / BLOCK],
                ElemType::from_scale_dtype(ScaleDtype::UE4M3),
            )),
            Some(PlainTensor::of_handle(tensor(
                client,
                &host.global.to_le_bytes(),
                [1, 1],
                F32,
            ))),
            scheme,
            Operand::of_scheme(&scheme, K, [N, K]).expect("one block along k"),
            [(N, n), (K, k)],
        );
        let x: Vec<u8> = host.x.iter().flat_map(|v| v.to_le_bytes()).collect();
        Device {
            weight,
            x: PlainTensor::of_handle(tensor(client, &x, [m, k], F16)),
            out: tensor(client, &vec![0u8; m * n * 2], [m, n], F16),
        }
    }

    fn out(&self) -> PlainTensor {
        PlainTensor::of_handle(self.out.clone())
    }
}

/// A routine at one setting, launched against one copy.
type Launch<'a> = Box<dyn Fn(&Device) -> Result<(), String> + Send + Sync + 'a>;

fn candidates<'a>(hw: &'a Hardware, client: &'a Client) -> Vec<(String, Launch<'a>)> {
    let mut out: Vec<(String, Launch<'a>)> = Vec::new();
    for settings in PlaneShuffleMatmulRoutine.packed_settings() {
        let launch: Launch<'a> = Box::new(move |d: &Device| {
            PlaneShuffleMatmulRoutine
                .launch_into(
                    hw,
                    client,
                    StoredTensor::Plain(d.x.clone()),
                    StoredTensor::Packed(d.weight.clone()),
                    d.out(),
                    &settings,
                )
                .map_err(|e| format!("{e:?}"))
        });
        out.push((settings.to_string(), launch));
    }
    for settings in CubeShuffleMatmulRoutine.packed_settings(hw) {
        let launch: Launch<'a> = Box::new(move |d: &Device| {
            CubeShuffleMatmulRoutine
                .launch_into(
                    hw,
                    client,
                    StoredTensor::Plain(d.x.clone()),
                    StoredTensor::Packed(d.weight.clone()),
                    d.out(),
                    &settings,
                )
                .map_err(|e| format!("{e:?}"))
        });
        out.push((settings.to_string(), launch));
    }
    out
}

fn read_f16(client: &Client, handle: &TensorHandle) -> Vec<f64> {
    client
        .read_one_unchecked(handle.handle.clone())
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16::from_le_bytes(*b).to_f64())
        .collect()
}

/// The largest error against the reference, relative to the output's magnitude.
fn worst_error(got: &[f64], host: &Host) -> f64 {
    let scale = host.expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(&host.expected)
        .map(|(g, w)| (g - w).abs() / scale)
        .fold(0.0, f64::max)
}

/// The device time of one launch, cycling through every copy so none is read warm.
fn time(client: &Client, data: &[Device], launch: &Launch<'_>) -> Duration {
    for d in data {
        launch(d).expect("launched once already");
    }
    block_on(client.sync()).expect("sync");
    let launches = (400 / data.len()).max(4) * data.len();
    let ((), duration) = client
        .profile(
            || {
                for i in 0..launches {
                    launch(&data[i % data.len()]).expect("launched once already");
                }
            },
            "nvfp4 gemv",
        )
        .expect("profile");
    let ticks = block_on(duration.resolve()).expect("device ticks");
    ticks.duration() / launches as u32
}

/// The host time of one launch's preparation and enqueue: past the device time, the device waits
/// on the host, and the timing is the host's.
fn host_time(client: &Client, data: &[Device], launch: &Launch<'_>) -> Duration {
    block_on(client.sync()).expect("sync");
    let start = std::time::Instant::now();
    for i in 0..data.len().max(32) {
        launch(&data[i % data.len()]).expect("launched once already");
    }
    let host = start.elapsed() / data.len().max(32) as u32;
    block_on(client.sync()).expect("sync");
    host
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let only_shape = std::env::var("NVFP4_SHAPE").ok();
    let only = std::env::var("NVFP4_ONLY").ok();
    // Qwen3-8B's projections as a decode step reads them, `[out, in]`.
    for (name, n, k) in [
        ("qkv", 6144, 4096),
        ("o", 4096, 4096),
        ("gate/up", 24576, 4096),
        ("down", 4096, 12288),
    ] {
        if only_shape.as_deref().is_some_and(|s| s != name) {
            continue;
        }
        let m = env_usize("NVFP4_M", 1);
        let host = Host::new(m, n, k);
        let bytes = n * k / 2 + n * k / BLOCK;
        let copies = ROTATION.div_ceil(bytes).max(2);
        let data: Vec<Device> = (0..copies)
            .map(|_| Device::new(&client, &host, m, n, k))
            .collect();
        println!("{name}: m {m}, n {n}, k {k}, {:.1} MB", bytes as f64 / 1e6);

        // Each setting that launches, and how far its output is from the reference.
        let mut served = Vec::new();
        for (label, launch) in candidates(&hw, &client) {
            if only.as_deref().is_some_and(|o| !label.contains(o)) {
                continue;
            }
            match launch(&data[0]) {
                Ok(()) => {
                    let got = read_f16(&client, &data[0].out);
                    let err = worst_error(&got, &host);
                    served.push((label, launch, err));
                }
                Err(why) => println!("  {label:<60} declined: {why}"),
            }
        }
        let mut best = vec![Duration::MAX; served.len()];
        for _ in 0..env_usize("NVFP4_ROUNDS", 3) {
            for (i, (_, launch, _)) in served.iter().enumerate() {
                best[i] = best[i].min(time(&client, &data, launch));
            }
        }
        for ((label, launch, err), t) in served.iter().zip(best) {
            let host = host_time(&client, &data, launch);
            println!(
                "  {label:<60} {:>7.1} GB/s  {:>7.1} µs  host {:>6.1} µs  err {err:.1e}",
                bytes as f64 / t.as_secs_f64() / 1e9,
                t.as_secs_f64() * 1e6,
                host.as_secs_f64() * 1e6
            );
        }
    }
}
