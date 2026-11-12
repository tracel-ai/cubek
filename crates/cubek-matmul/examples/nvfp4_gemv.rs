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
//! `NVFP4_ONLY` keeps the settings whose name contains it or any of its `|`-separated parts,
//! `NVFP4_LAYOUT` keeps the weight in `rows` or in `tiles`, the tile order a load stores it in on
//! this device (both). `NVFP4_STAGGER` lays the tiles' strips back to back (`0`) or an odd
//! number of that many bytes apart, in place of a transaction.
//! `NVFP4_PROFILE` launches each setting once a copy and times nothing, for a profiler.
//! `NVFP4_N` and `NVFP4_K` time one shape of their own in place of the four.
//! `NVFP4_EXPLORE` races, in place of the cube shuffle's packed settings, a grid of cube shuffles
//! reading the weight by block: columns a unit, planes along `K`, cubes a compute unit, and how
//! a plane's units split `K`; `NVFP4_EXPLORE=planes` also races a grid of plane shuffles.

use std::time::Duration;

use cubecl::client::Client;
use cubecl::future::block_on;
use cubecl::ir::{ElemType, FloatKind, UIntKind};

use cubecl::quant::scheme::ScaleDtype;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Field;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, MatmulRoutine,
    PlaneShuffleMatmulRoutine,
};
use cubek_matmul::tiled::storage;
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order;
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
    fn new(
        client: &Client,
        hw: &Hardware,
        host: &Host,
        (m, n, k): (usize, usize, usize),
        layout: Layout,
    ) -> Self {
        let scheme = nvfp4_scheme();
        let words: Vec<u8> = host.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales: Vec<u8> = host
            .scale_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let global = Some(PlainTensor::of_handle(tensor(
            client,
            &host.global.to_le_bytes(),
            [1, 1],
            F32,
        )));
        if layout == Layout::Tiles {
            let x: Vec<u8> = host.x.iter().flat_map(|v| v.to_le_bytes()).collect();
            return Device {
                weight: in_tile_order(client, hw, &words, &scales, global, (n, k)),
                x: PlainTensor::of_handle(tensor(client, &x, [m, k], F16)),
                out: tensor(client, &vec![0u8; m * n * 2], [m, n], F16),
            };
        }
        let weight = PackedTensor::new(
            PlainTensor::of_handle(tensor(client, &words, [n, k / CODES_PER_WORD], WORD)),
            // A byte a block, as a checkpoint stores them: four share a word in memory.
            PlainTensor::of_handle(tensor(
                client,
                &scales,
                [n, k / BLOCK],
                ElemType::from_scale_dtype(ScaleDtype::UE4M3),
            )),
            global,
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

/// Where the weight's words and block scales lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Each column's words and scales down `k`, as a checkpoint stores them.
    Rows,
    /// The tile order a load stores the weight in on this device, its scales beside it, as
    /// Metabolic's `into_tile_order` lays it out.
    Tiles,
}

/// The weight in `words` (`[n, k / 8]`) and `scales` (`[n, k / 16]` bytes) laid out in the tile
/// order a load stores it in on `hw`, at the `64 × 16` tile a decode step's packed weight is
/// stored in on an RDNA GPU.
fn in_tile_order(
    client: &Client,
    hw: &Hardware,
    words: &[u8],
    scales: &[u8],
    global: Option<PlainTensor>,
    (n, k): (usize, usize),
) -> PackedTensor {
    use cubecl::prelude::TensorBinding;
    use cubecl::zspace::metadata::Metadata;
    use cubek_matmul::kernels::tensor::PackedTensor as Packed;

    // The same bytes stated over `[k, n]`, its words down `k`: the statement tile order stores.
    let scheme = tile_order::nvfp4_scheme();
    // `NVFP4_STAGGER=0` lays the strips back to back, as a load did before it staggered them;
    // `NVFP4_STAGGER=<bytes>` staggers them by an odd number of that many bytes in place of a
    // transaction.
    let laid = || {
        tile_order::TileOrder::laid(&scheme, storage::tile_words(hw), (64, 16))
            .expect("NVFP4 is tile order")
    };
    let order = match env_usize("NVFP4_STAGGER", usize::MAX) {
        0 => laid().for_matrix(k, n),
        usize::MAX => {
            storage::tile_order_on(hw, &scheme, (64, 16), (k, n)).expect("NVFP4 is tile order")
        }
        bytes => laid()
            .staggered(
                cubecl::tile::layout::Stagger::new(bytes * 8, 8)
                    .expect("a byte divides the stagger"),
            )
            .for_matrix(k, n),
    };
    let words_per_load = hw.load_values(WORD);
    let blocks = k / BLOCK;
    let bound = |handle: cubecl::server::Handle, meta: Metadata| TensorBinding {
        handle: handle.binding(),
        strides: meta.strides,
        shape: meta.shape,
        tiling: meta.tiling,
    };
    // The values as the kernel names them, `[n, k/block, block]`, counted in values over words.
    let values = storage::tile(
        client,
        bound(
            client.create_from_slice(words),
            Metadata::new([n, blocks, BLOCK], [k, BLOCK, 1]),
        ),
        &tile_order::values_axes(N),
        WORD,
        Some(Field::from(scheme.value)),
        order.values_storage(N, words_per_load),
    )
    .expect("the values tile");
    // The scales, `[n, k/block]`, a byte each, counted in scales over words.
    let tiled_scales = storage::tile(
        client,
        bound(
            client.create_from_slice(scales),
            Metadata::new([n, blocks], [blocks, 1]),
        ),
        &tile_order::scales_axes(N),
        WORD,
        Some(Field::of_scale(ScaleDtype::UE4M3)),
        order
            .scales_storage(N, order.rows_a_load(N, words_per_load))
            .expect("block by block, the scales lie in tiles"),
    )
    .expect("the scales tile");
    let matrix = tile_order::matrix_values(&values.metadata);
    let sized = [(K, k), (N, n)];
    let operand = Operand::of_scheme(&scheme, K, [K, N]).expect("one block along k");
    let operand = Packed::read_in_tiles(hw, sized, operand, &matrix).expect("read in its tiles");
    Packed::new(
        PlainTensor::new(bound(values.handle, matrix), WORD),
        PlainTensor::new(bound(tiled_scales.handle, *tiled_scales.metadata), WORD),
        global,
        scheme,
        operand,
        sized,
    )
}

/// A routine at one setting, launched against one copy.
type Launch<'a> = Box<dyn Fn(&Device) -> Result<(), String> + Send + Sync + 'a>;

fn candidates<'a>(hw: &'a Hardware, client: &'a Client) -> Vec<(String, Launch<'a>)> {
    let mut out: Vec<(String, Launch<'a>)> = Vec::new();
    let plane_settings = match std::env::var("NVFP4_EXPLORE").as_deref() {
        Ok("planes") => explored_planes(),
        _ => PlaneShuffleMatmulRoutine.packed_settings(),
    };
    for settings in plane_settings {
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
    let cube_settings = match std::env::var_os("NVFP4_EXPLORE") {
        Some(_) => explored(),
        None => CubeShuffleMatmulRoutine.packed_settings(hw),
    };
    for settings in cube_settings {
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

/// The plane shuffles `NVFP4_EXPLORE=planes` races beside the cube shuffles' grid: columns a
/// unit, planes a cube and planes a compute unit, under each split of `K` a plane's units take.
fn explored_planes() -> Vec<cubek_matmul::kernels::PlaneShuffleMatmulRoutineSettings> {
    use core::num::NonZeroUsize;
    use cubek_microkernel::matmul::plane::PlaneShuffleMicrokernelSettings;
    let mut out = Vec::new();
    for columns in [1, 2, 4] {
        for split in [0, 1, 2] {
            let plane = PlaneShuffleMicrokernelSettings::stepping(columns, None);
            let microkernel = match split {
                0 => plane,
                1 => plane.one_transaction(),
                _ => plane.one_tile(),
            };
            for planes in [1, 2, 4, 8] {
                for planes_per_worker in [None, NonZeroUsize::new(16), NonZeroUsize::new(32)] {
                    out.push(cubek_matmul::kernels::PlaneShuffleMatmulRoutineSettings {
                        microkernel,
                        planes: NonZeroUsize::new(planes).unwrap(),
                        planes_per_worker,
                    });
                }
            }
        }
    }
    out
}

/// The cube shuffles `NVFP4_EXPLORE` races: each column count a unit, planes along `K`, cubes a
/// compute unit, under either split of `K` a plane's units take, reading the weight by block, and
/// under one transaction also as stored.
fn explored() -> Vec<CubeShuffleMatmulRoutineSettings> {
    use core::num::NonZeroUsize;
    use cubek_microkernel::matmul::cube::{
        CubeShuffleMicrokernelSettings, CubeShufflePlaneSettings,
    };
    use cubek_microkernel::matmul::plane::PlaneShuffleMicrokernelSettings;
    let mut out = Vec::new();
    for one_transaction in [false, true] {
        for columns in [1, 2, 4, 8] {
            for planes in [16, 32] {
                for cubes in [
                    None,
                    NonZeroUsize::new(1),
                    NonZeroUsize::new(2),
                    NonZeroUsize::new(4),
                ] {
                    let plane = PlaneShuffleMicrokernelSettings::stepping(columns, None);
                    let plane = match one_transaction {
                        true => plane.one_transaction(),
                        false => plane,
                    };
                    let microkernel = CubeShuffleMicrokernelSettings::new(
                        CubeShufflePlaneSettings::Shuffle(plane),
                        planes,
                    );
                    out.push(CubeShuffleMatmulRoutineSettings {
                        microkernel: microkernel.by_block(),
                        cubes_per_worker: cubes,
                    });
                    // Under one transaction, also each load of the weight as stored, a word of
                    // scales a turn.
                    if one_transaction {
                        out.push(CubeShuffleMatmulRoutineSettings {
                            microkernel,
                            cubes_per_worker: cubes,
                        });
                    }
                }
            }
        }
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

mod probe {
    use cubecl::prelude::*;
    use cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};
    use cubecl::{self as cubecl, cube};

    /// Every unit sums `per_unit` vectors of `input`, its own position and every `output.shape()`
    /// after it, and writes the sum: a read of the whole input and nothing else, the most a kernel
    /// reading these bytes could reach.
    #[cube(launch, address_type = "dynamic")]
    pub fn read_every_vector<E: Numeric, N: Size>(
        input: LinearView<'_, Vector<E, N>>,
        mut output: LinearViewMut<'_, Vector<E, N>>,
        #[comptime] per_unit: usize,
        #[define(E)] _dtype: ElemType,
    ) {
        if !output.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let units = output.shape();
        let mut sum = Vector::<E, N>::cast_from(0u32);
        #[unroll]
        for i in 0..per_unit {
            let vector = ABSOLUTE_POS + i * units;
            if input.is_in_bounds(vector) {
                sum += input.read(vector);
            }
        }
        output.write(ABSOLUTE_POS, sum);
    }
}

/// The best read of `bytes` bytes, cold, over copies laid end to end past the last-level cache,
/// at each count of 16-byte vectors a unit: the ceiling a product reading them meets.
fn read_ceiling(client: &Client, bytes: usize) -> (f64, usize) {
    use cubecl::prelude::*;
    use cubecl::std::tensor::layout::linear::linear_view;
    let words = bytes.div_ceil(16) * 4;
    let copies = ROTATION.div_ceil(bytes).max(2);
    let inputs: Vec<TensorHandle> = (0..copies)
        .map(|_| tensor(client, &vec![1u8; words * 4], [1, words], WORD))
        .collect();
    let vector_size = 4;
    let mut best = (0.0, 0);
    for per_unit in [1, 2, 4, 8, 16] {
        let units = (words / vector_size).div_ceil(per_unit);
        let output = tensor(
            client,
            &vec![0u8; units * vector_size * 4],
            [1, units * vector_size],
            WORD,
        );
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(units.div_ceil(256) as u32, 1, 1);
        let launch = |input: &TensorHandle| {
            probe::read_every_vector::launch(
                client,
                cube_count.clone(),
                cube_dim,
                input.required_address_type(),
                vector_size,
                linear_view(input.clone().binding()),
                linear_view(output.clone().binding()),
                per_unit,
                WORD,
            )
        };
        for input in &inputs {
            launch(input);
        }
        block_on(client.sync()).expect("sync");
        let launches = (400 / copies).max(4) * copies;
        let mut fastest = Duration::MAX;
        for _ in 0..env_usize("NVFP4_ROUNDS", 3) {
            let ((), duration) = client
                .profile(
                    || {
                        for i in 0..launches {
                            launch(&inputs[i % copies]);
                        }
                    },
                    "read ceiling",
                )
                .expect("profile");
            let ticks = block_on(duration.resolve()).expect("device ticks");
            fastest = fastest.min(ticks.duration() / launches as u32);
        }
        let gbs = bytes as f64 / fastest.as_secs_f64() / 1e9;
        if gbs > best.0 {
            best = (gbs, per_unit);
        }
    }
    best
}

fn main() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let only_shape = std::env::var("NVFP4_SHAPE").ok();
    let only = std::env::var("NVFP4_ONLY").ok();
    // Qwen3-8B's projections as a decode step reads them, `[out, in]`.
    // `NVFP4_N` and `NVFP4_K` together time one shape of their own instead, named `custom`.
    let custom = env_usize("NVFP4_N", 0) * env_usize("NVFP4_K", 0) > 0;
    let shapes = match custom {
        true => vec![("custom", env_usize("NVFP4_N", 0), env_usize("NVFP4_K", 0))],
        false => vec![
            ("qkv", 6144, 4096),
            ("o", 4096, 4096),
            ("gate/up", 24576, 4096),
            ("down", 4096, 12288),
        ],
    };
    for (name, n, k) in shapes {
        if only_shape.as_deref().is_some_and(|s| s != name) {
            continue;
        }
        let m = env_usize("NVFP4_M", 1);
        let host = Host::new(m, n, k);
        let bytes = n * k / 2 + n * k / BLOCK;
        let copies = ROTATION.div_ceil(bytes).max(2);
        let profile = std::env::var_os("NVFP4_PROFILE").is_some();
        if !profile {
            let (ceiling, per_unit) = read_ceiling(&client, bytes);
            println!("{name} ceiling: {ceiling:.1} GB/s ({per_unit} vectors a unit)");
        }
        for layout in [Layout::Rows, Layout::Tiles] {
            let wanted = std::env::var("NVFP4_LAYOUT").ok();
            if wanted
                .as_deref()
                .is_some_and(|w| !format!("{layout:?}").eq_ignore_ascii_case(w))
            {
                continue;
            }
            let data: Vec<Device> = (0..copies)
                .map(|_| Device::new(&client, &hw, &host, (m, n, k), layout))
                .collect();
            println!(
                "{name} in {layout:?}: m {m}, n {n}, k {k}, {:.1} MB",
                bytes as f64 / 1e6
            );

            // Each setting that launches, and how far its output is from the reference.
            let mut served = Vec::new();
            for (label, launch) in candidates(&hw, &client) {
                if only
                    .as_deref()
                    .is_some_and(|o| !o.split('|').any(|o| label.contains(o)))
                {
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
            // `NVFP4_PROFILE`: each served setting launched once a copy, cold, and nothing timed:
            // the launches a profiler counts.
            if profile {
                for (label, launch, _) in &served {
                    for d in &data {
                        launch(d).expect("launched once already");
                    }
                    block_on(client.sync()).expect("sync");
                    println!("  {label} profiled over {} copies", data.len());
                }
                continue;
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
}
