//! A prompt's projections with both inputs block-scaled, as W8A8 and W4A8 run them: MXFP8
//! activations against an MXFP8 or an MXFP4 weight, beside the same product in `f16`, at Qwen3.8
//! 27B's projections. Every setting of the staged routines that reads both inputs as stored is
//! timed, each checked against an `f64` reference on a sample of the output's cells.
//!
//! On a device whose 8-bit float instruction multiplies the codes (RDNA4's quantized MMA), a
//! packed product contracts through it, the weight's `e2m1` codes widened to `e4m3`; elsewhere it
//! decodes its inputs into a 16-bit instruction's operands.
//!
//! Every launch reads its operands cold (a rotation through 512 MB of copies), in device time, the
//! best of interleaved rounds, in TFLOP/s of the product's `2 m n k`.
//!
//! ```sh
//! cargo run --release -p cubek-matmul --example block_scaled_prefill --features cubecl/hip
//! ```
//!
//! Both packed inputs are stored in tile order, the tile a load stores each in on this device
//! (the step a load stores it in, `MatmulStep::stored_on`), as a model lays its weights and
//! a quantizer its activations; `BS_LAYOUT=rows` keeps them in the rows a checkpoint packs.
//!
//! `BS_M` sets the rows (512), `BS_ROUNDS` the rounds (3), `BS_SHAPE` keeps one projection,
//! `BS_FORMAT` keeps the formats whose name contains it (`f16`, `mxfp8`, `w4a8`), `BS_ONLY` keeps
//! the settings whose name contains it. `BS_N` and `BS_K` time one shape of their own.

use std::time::Duration;

use cubecl::client::Client;
use cubecl::future::block_on;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Field;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    MatmulRoutine,
};
use cubek_matmul::tiled::storage;
use cubek_microkernel::matmul::{MatmulOperand, MatmulStep};
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order;
use cubek_std::matmul_dimensions::{K, M, N};
use cubek_std::routine::Routine;
use half::f16;

/// Values of `k` one block scale covers, in both formats.
const BLOCK: usize = 32;
const WORD: ElemType = ElemType::UInt(UIntKind::U32);
const F16: ElemType = ElemType::Float(FloatKind::F16);
/// Twice the largest last-level cache in this class.
const ROTATION: usize = 512 << 20;
/// Output cells the reference checks.
const SAMPLES: usize = 512;

/// `e2m1`'s magnitudes, by their three low bits.
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// An `e4m3` byte's value (OCP: bias 7, no infinity).
fn e4m3(byte: u8) -> f64 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let (exp, mant) = ((byte >> 3) & 0xF, byte & 0x7);
    sign * match exp {
        0 => (mant as f64 / 8.0) * 2f64.powi(-6),
        _ => (1.0 + mant as f64 / 8.0) * 2f64.powi(exp as i32 - 7),
    }
}

/// A `ue8m0` byte's value.
fn ue8m0(byte: u8) -> f64 {
    2f64.powi(byte as i32 - 127)
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

/// How an input is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    F16,
    Mxfp8,
    Mxfp4,
}

impl Format {
    fn scheme(self) -> Option<QuantScheme> {
        let value = match self {
            Format::F16 => return None,
            Format::Mxfp8 => QuantValue::E4M3,
            Format::Mxfp4 => QuantValue::E2M1,
        };
        Some(
            QuantScheme::default()
                .with_value(value)
                .per_block([BLOCK as u8], ScaleDtype::UE8M0)
                .with_store(QuantStore::PackedU32(0)),
        )
    }

    fn codes_per_word(self) -> usize {
        match self {
            Format::Mxfp8 => 4,
            Format::Mxfp4 => 8,
            Format::F16 => 1,
        }
    }

    /// A random code's byte (or nibble): never `e4m3`'s NaN, so a product never overflows `f16`.
    fn code(self, rng: &mut Rng) -> u8 {
        match self {
            // Magnitudes up to 2^2, either sign.
            Format::Mxfp8 => {
                let magnitude = (rng.next() % 0x48) as u8;
                magnitude | if rng.next() & 1 == 0 { 0 } else { 0x80 }
            }
            Format::Mxfp4 => (rng.next() % 16) as u8,
            Format::F16 => 0,
        }
    }

    fn value(self, code: u8) -> f64 {
        match self {
            Format::Mxfp8 => e4m3(code),
            Format::Mxfp4 => {
                let magnitude = E2M1[(code & 7) as usize];
                if code & 8 != 0 { -magnitude } else { magnitude }
            }
            Format::F16 => 0.0,
        }
    }
}

/// One input on the host: `rows × k` codes along `k`, a `ue8m0` scale every block of them, and
/// their values.
struct HostInput {
    codes: Vec<u8>,
    scales: Vec<u8>,
    values: Vec<f64>,
}

impl HostInput {
    fn new(format: Format, rows: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let quantized = match format {
            Format::F16 => Format::Mxfp8,
            other => other,
        };
        let codes: Vec<u8> = (0..rows * k).map(|_| quantized.code(&mut rng)).collect();
        // Scales from 2^-2 to 2^1.
        let scales: Vec<u8> = (0..rows * k / BLOCK)
            .map(|_| (125 + rng.next() % 4) as u8)
            .collect();
        let values = codes
            .iter()
            .enumerate()
            .map(|(at, &code)| quantized.value(code) * ue8m0(scales[at / BLOCK]))
            .collect();
        HostInput {
            codes,
            scales,
            values,
        }
    }

    /// The codes packed into words along `k`, the first in the lowest bits.
    fn words(&self, format: Format) -> Vec<u8> {
        let per_word = format.codes_per_word();
        let bits = 32 / per_word;
        let mask = (1u32 << bits) - 1;
        self.codes
            .chunks(per_word)
            .flat_map(|w| {
                w.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &c)| acc | ((c as u32 & mask) << (bits * j)))
                    .to_le_bytes()
            })
            .collect()
    }
}

fn tensor(client: &Client, bytes: &[u8], shape: [usize; 2], elem: ElemType) -> TensorHandle {
    TensorHandle::new_contiguous(
        Shape::from(shape.to_vec()),
        client.create_from_slice(bytes),
        elem,
    )
}

/// Where a packed input's words and block scales lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Each row's words and scales down `k`, as a checkpoint stores them.
    Rows,
    /// The tile order a load stores the input in on this device, its scales beside it.
    Tiles,
}

/// `input`, `rows × k`, on the device as `format` stores it: `[rows, k]` for the lhs and for a
/// packed weight (`[n, k]`, words along `k`), in rows or in tile order as `layout` says;
/// `[k, n]` row-major for an `f16` weight.
fn stored(
    client: &Client,
    hw: &Hardware,
    input: &HostInput,
    format: Format,
    layout: Layout,
    (across, rows, k): (cubecl::tile::Dimension, usize, usize),
) -> StoredTensor {
    let Some(scheme) = format.scheme() else {
        let halves = |values: Vec<f64>| -> Vec<u8> {
            values
                .into_iter()
                .flat_map(|v| f16::from_f64(v).to_le_bytes())
                .collect()
        };
        return match across == M {
            true => StoredTensor::Plain(PlainTensor::of_handle(tensor(
                client,
                &halves(input.values.clone()),
                [rows, k],
                F16,
            ))),
            false => {
                let transposed = (0..k * rows)
                    .map(|at| input.values[(at % rows) * k + at / rows])
                    .collect();
                StoredTensor::Plain(PlainTensor::of_handle(tensor(
                    client,
                    &halves(transposed),
                    [k, rows],
                    F16,
                )))
            }
        };
    };
    if layout == Layout::Tiles {
        return StoredTensor::Packed(in_tile_order(client, hw, input, format, (across, rows, k)));
    }
    StoredTensor::Packed(PackedTensor::new(
        PlainTensor::of_handle(tensor(
            client,
            &input.words(format),
            [rows, k / format.codes_per_word()],
            WORD,
        )),
        PlainTensor::of_handle(tensor(
            client,
            &input.scales,
            [rows, k / BLOCK],
            ElemType::from_scale_dtype(ScaleDtype::UE8M0),
        )),
        None,
        scheme,
        Operand::of_scheme(&scheme, K, [across, K]).expect("one block along k"),
        [(across, rows), (K, k)],
    ))
}

/// `input`, packed as `format` stores it, laid out in the tile order a load stores it in on `hw`
/// ([`MatmulStep::stored_on`], [`storage::tile_order_on`]): the values as the kernel names them,
/// `[rows, k/block, block]`, and the block scales in tiles of their own where the order keeps
/// them there, else in their rows.
fn in_tile_order(
    client: &Client,
    hw: &Hardware,
    input: &HostInput,
    format: Format,
    (across, rows, k): (cubecl::tile::Dimension, usize, usize),
) -> PackedTensor {
    use cubecl::prelude::TensorBinding;
    use cubecl::zspace::metadata::Metadata;

    let in_rows = format.scheme().expect("a packed format");
    let input_of = match across == M {
        true => MatmulOperand::Lhs,
        false => MatmulOperand::Rhs,
    };
    // The step a load stores the input in, as `storage::stored_tile` takes it for a weight: under
    // the largest instruction that contracts it, the other input's codes `e4m3` in every packed
    // product timed here.
    let contracted = ElemType::Float(FloatKind::E4M3);
    let read = Operand::of_scheme(&in_rows, K, [across, K]).expect("one block along k");
    let values = cubek_std::value_element(in_rows.value).expect("float codes");
    let step = MatmulStep::stored_on(hw, input_of, read, values, contracted)
        .expect("the device contracts the input");
    let tile = (step.along(K), step.along(across));
    // The same bytes stated over `[k, rows]`, its words down `k`: the statement tile order stores.
    let scheme = QuantScheme::default()
        .with_value(in_rows.value)
        .per_block([BLOCK as u8, 1], ScaleDtype::UE8M0)
        .with_store(QuantStore::PackedU32(1));
    let order =
        storage::tile_order_on(hw, &scheme, tile, (k, rows)).expect("tile order stores the input");
    let words_per_load = hw.load_values(WORD);
    let blocks = k / BLOCK;
    let bound = |handle: cubecl::server::Handle, meta: Metadata| TensorBinding {
        handle: handle.binding(),
        strides: meta.strides,
        shape: meta.shape,
        tiling: meta.tiling,
    };
    let values = storage::tile(
        client,
        bound(
            client.create_from_slice(&input.words(format)),
            Metadata::new([rows, blocks, BLOCK], [k, BLOCK, 1]),
        ),
        &tile_order::values_dimensions(across),
        WORD,
        Some(Field::from(scheme.value)),
        order.values_storage(across, words_per_load),
    )
    .expect("the values tile");
    let scales = match order.scales_storage(across, order.rows_a_load(across, words_per_load)) {
        Some(laid) => {
            let tiled = storage::tile(
                client,
                bound(
                    client.create_from_slice(&input.scales),
                    Metadata::new([rows, blocks], [blocks, 1]),
                ),
                &tile_order::scales_dimensions(across),
                WORD,
                Some(Field::of_scale(ScaleDtype::UE8M0)),
                laid,
            )
            .expect("the scales tile");
            PlainTensor::new(bound(tiled.handle, *tiled.metadata), WORD)
        }
        None => PlainTensor::of_handle(tensor(
            client,
            &input.scales,
            [rows, blocks],
            ElemType::from_scale_dtype(ScaleDtype::UE8M0),
        )),
    };
    let matrix = tile_order::matrix_values(&values.metadata);
    let sized = [(K, k), (across, rows)];
    let operand = Operand::of_scheme(&scheme, K, [K, across]).expect("one block along k");
    let operand =
        PackedTensor::read_in_tiles(hw, sized, operand, &matrix).expect("read in its tiles");
    PackedTensor::new(
        PlainTensor::new(bound(values.handle, matrix), WORD),
        scales,
        None,
        scheme,
        operand,
        sized,
    )
}

/// One copy of the operands on the device.
struct Device {
    lhs: StoredTensor,
    rhs: StoredTensor,
    out: TensorHandle,
}

impl Device {
    fn out(&self) -> PlainTensor {
        PlainTensor::of_handle(self.out.clone())
    }
}

/// A routine at one setting, launched against one copy.
type Launch<'a> = Box<dyn Fn(&Device) -> Result<(), String> + Send + Sync + 'a>;

fn launch<'a, R, M2, SM>(
    hw: &'a Hardware,
    client: &'a Client,
    routine: R,
    settings: R::Settings,
) -> Launch<'a>
where
    R: MatmulRoutine<M2, SM> + Send + Sync + 'a,
    R::Settings: Send + Sync + 'a,
    M2: cubek_microkernel::matmul::MatmulMicrokernel,
{
    Box::new(move |d: &Device| {
        routine
            .launch_into(hw, client, d.lhs.clone(), d.rhs.clone(), d.out(), &settings)
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
    })
}

/// The settings timed for a product of `lhs` by `rhs`: every setting the cube staged routine
/// lists for a float product, and for a packed one its packed settings, its default, and the
/// copy engine's block-scaled settings.
fn candidates<'a>(hw: &'a Hardware, client: &'a Client, packed: bool) -> Vec<(String, Launch<'a>)> {
    let mut out: Vec<(String, Launch<'a>)> = Vec::new();
    if !packed {
        for settings in <CubeStagedMatmulRoutine as Routine>::settings(hw) {
            out.push((
                format!("staged {settings}"),
                launch(hw, client, CubeStagedMatmulRoutine, settings),
            ));
        }
        return out;
    }
    let staged = [
        vec![CubeStagedMatmulRoutineSettings::DEFAULT],
        CubeStagedMatmulRoutine.packed_settings(hw),
    ]
    .concat();
    for settings in staged {
        out.push((
            format!("staged {settings}"),
            launch(hw, client, CubeStagedMatmulRoutine, settings),
        ));
    }
    for settings in CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(hw) {
        out.push((
            format!("async-copy {settings}"),
            launch(hw, client, CubeStagedAsyncCopyMatmulRoutine, settings),
        ));
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

/// The sampled cells of the product on the host, each `(cell, value)`.
fn reference(
    lhs: &HostInput,
    rhs: &HostInput,
    (m, n, k): (usize, usize, usize),
) -> Vec<(usize, f64)> {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    (0..SAMPLES)
        .map(|_| {
            let cell = (rng.next() as usize) % (m * n);
            let (row, col) = (cell / n, cell % n);
            let dot = (0..k)
                .map(|l| lhs.values[row * k + l] * rhs.values[col * k + l])
                .sum();
            (cell, dot)
        })
        .collect()
}

/// The largest error on the sampled cells, relative to their largest magnitude.
fn worst_error(got: &[f64], expected: &[(usize, f64)]) -> f64 {
    let scale = expected.iter().fold(1.0f64, |m, (_, v)| m.max(v.abs()));
    expected
        .iter()
        .map(|(cell, want)| (got[*cell] - want).abs() / scale)
        .fold(0.0, f64::max)
}

/// The device time of one launch, cycling through every copy so none is read warm.
fn time(client: &Client, data: &[Device], launch: &Launch<'_>) -> Duration {
    for d in data {
        launch(d).expect("launched once already");
    }
    block_on(client.sync()).expect("sync");
    let launches = (64 / data.len()).max(2) * data.len();
    let ((), duration) = client
        .profile(
            || {
                for i in 0..launches {
                    launch(&data[i % data.len()]).expect("launched once already");
                }
            },
            "block-scaled prefill",
        )
        .expect("profile");
    let ticks = block_on(duration.resolve()).expect("device ticks");
    ticks.duration() / launches as u32
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
    let only_shape = std::env::var("BS_SHAPE").ok();
    let only_format = std::env::var("BS_FORMAT").ok();
    let only = std::env::var("BS_ONLY").ok();
    let m = env_usize("BS_M", 512);
    let layout = match std::env::var("BS_LAYOUT").as_deref() {
        Ok("rows") => Layout::Rows,
        _ => Layout::Tiles,
    };
    // Qwen3.8 27B's projections as a prompt reads them, `[out, in]`.
    let shapes = match (env_usize("BS_N", 0), env_usize("BS_K", 0)) {
        (n, k) if n > 0 && k > 0 => vec![("custom", n, k)],
        _ => vec![
            ("qkv", 8192, 5120),
            ("o", 5120, 6144),
            ("gate/up", 34816, 5120),
            ("down", 5120, 17408),
        ],
    };
    let formats = [
        ("f16", Format::F16, Format::F16),
        ("mxfp8", Format::Mxfp8, Format::Mxfp8),
        ("w4a8", Format::Mxfp8, Format::Mxfp4),
    ];
    for (name, n, k) in shapes {
        if only_shape.as_deref().is_some_and(|s| s != name) {
            continue;
        }
        let flops = 2.0 * (m * n * k) as f64;
        for (label, lhs_format, rhs_format) in formats {
            if only_format.as_deref().is_some_and(|f| !label.contains(f)) {
                continue;
            }
            let lhs = HostInput::new(lhs_format, m, k, 0x9E37_79B9_7F4A_7C15);
            let rhs = HostInput::new(rhs_format, n, k, 0xC2B2_AE3D_27D4_EB4F);
            let expected = reference(&lhs, &rhs, (m, n, k));
            let bytes = |format: Format, rows: usize| match format {
                Format::F16 => rows * k * 2,
                packed => rows * k / packed.codes_per_word() * 4 + rows * k / BLOCK,
            };
            let copy_bytes = bytes(lhs_format, m) + bytes(rhs_format, n) + m * n * 2;
            let copies = ROTATION.div_ceil(copy_bytes).clamp(2, 8);
            let data: Vec<Device> = (0..copies)
                .map(|_| Device {
                    lhs: stored(&client, &hw, &lhs, lhs_format, layout, (M, m, k)),
                    rhs: stored(&client, &hw, &rhs, rhs_format, layout, (N, n, k)),
                    out: tensor(&client, &vec![0u8; m * n * 2], [m, n], F16),
                })
                .collect();
            println!("{name} {label} in {layout:?}: m {m}, n {n}, k {k}");
            let mut served = Vec::new();
            for (setting, launch) in candidates(&hw, &client, lhs_format != Format::F16) {
                if only.as_deref().is_some_and(|o| !setting.contains(o)) {
                    continue;
                }
                match launch(&data[0]) {
                    Ok(()) => {
                        let got = read_f16(&client, &data[0].out);
                        served.push((setting, launch, worst_error(&got, &expected)));
                    }
                    Err(why) => println!("  {setting:<70} declined: {why}"),
                }
            }
            let mut best = vec![Duration::MAX; served.len()];
            for _ in 0..env_usize("BS_ROUNDS", 3) {
                for (i, (_, launch, _)) in served.iter().enumerate() {
                    best[i] = best[i].min(time(&client, &data, launch));
                }
            }
            let mut fastest: Option<(f64, &str)> = None;
            for ((setting, _, err), t) in served.iter().zip(&best) {
                let tflops = flops / t.as_secs_f64() / 1e12;
                println!(
                    "  {setting:<70} {tflops:>7.1} TFLOP/s  {:>8.1} µs  err {err:.1e}",
                    t.as_secs_f64() * 1e6
                );
                if *err < 1e-2 && fastest.is_none_or(|(f, _)| tflops > f) {
                    fastest = Some((tflops, setting));
                }
            }
            if let Some((tflops, setting)) = fastest {
                println!("{name} {label} best: {tflops:.1} TFLOP/s ({setting})");
            }
        }
    }
}
