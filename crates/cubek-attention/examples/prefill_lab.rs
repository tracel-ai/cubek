//! LAB: times prefill attention routines at chosen settings on one chunk's problem.
//!
//! `cargo run --release --features cubecl/cuda --example prefill_lab -- <heads> <kv_heads> <queries> <keys> <suite>`
//!
//! A chunk is `queries` positions placed last against `keys` cached positions, causal, head dim
//! 128, `f16`. Every setting's output is checked against the suite's first. One CSV line per
//! setting on stdout.

use core::num::NonZeroUsize;
use cubecl::__private::Runtime as _;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::{Shape, metadata::Metadata};
use cubecl::{TestRuntime, future};
use cubek_attention::kernels::*;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernelSettings, CubeAttentionPlaneSettings,
};
use cubek_microkernel::attention::hypercube::HypercubeAttentionMicrokernelSettings;
use cubek_microkernel::attention::plane::PlaneFragmentAttentionMicrokernelSettings;
use cubek_microkernel::matmul::cube::CubeStagedDelivery;
use cubek_std::Hardware;
use cubek_std::microkernel::{StageLayout, StageRing};
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const DIM: usize = 128;
const F16: ElemType = ElemType::Float(FloatKind::F16);

fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn upload(client: &Client, shape: &[usize], bytes: &[u8], elem: ElemType) -> PlainTensor {
    let handle = client.create_from_slice(bytes);
    let meta = Metadata::new(Shape::from(shape.to_vec()), contiguous(shape));
    PlainTensor::of_handle(TensorHandle::from_metadata(handle, meta, elem))
}

/// Deterministic values in [-1, 1) as `f16`.
fn random_f16(client: &Client, shape: &[usize], seed: u64) -> PlainTensor {
    let count: usize = shape.iter().product();
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bytes: Vec<u8> = (0..count)
        .flat_map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
            half::f16::from_f32(unit * 2.0 - 1.0).to_le_bytes()
        })
        .collect();
    upload(client, shape, &bytes, F16)
}

struct Lab {
    client: Client,
    hw: Hardware,
    query: PlainTensor,
    key: PlainTensor,
    value: PlainTensor,
    lengths: PlainTensor,
    scoring: AttentionScoring,
    flops: f64,
    reference: Option<Vec<f32>>,
}

impl Lab {
    fn new(heads: usize, kv_heads: usize, queries: usize, keys: usize) -> Self {
        let client = TestRuntime::client(&Default::default());
        let hw = Hardware::from_client(&client);
        let query = random_f16(&client, &[1, heads, queries, DIM], 1);
        let key = random_f16(&client, &[1, kv_heads, keys, DIM], 2);
        let value = random_f16(&client, &[1, kv_heads, keys, DIM], 3);
        let lengths = upload(
            &client,
            &[1],
            &(keys as u32).to_le_bytes(),
            ElemType::UInt(UIntKind::U32),
        );
        // `CUBEK_LAB_NONCAUSAL`: every query sees every key.
        let causal = std::env::var("CUBEK_LAB_NONCAUSAL").is_err();
        let visibility = AttentionVisibility::new(causal, None, QueryPlacement::Last);
        let scoring = AttentionScoring::standard(visibility, DIM);
        // Query i sees the `keys - queries` before the chunk and the chunk up to itself.
        let seen = match causal {
            true => queries * (keys - queries) + queries * (queries + 1) / 2,
            false => queries * keys,
        };
        let flops = 4.0 * (DIM * heads * seen) as f64;
        Self {
            client,
            hw,
            query,
            key,
            value,
            lengths,
            scoring,
            flops,
            reference: None,
        }
    }

    fn tensors(&self) -> AttentionTensors {
        AttentionTensors::new(
            &self.client,
            self.query.clone(),
            self.key.clone(),
            self.value.clone(),
            self.lengths.clone(),
            self.scoring,
        )
        .expect("the lab's tensors are admitted")
    }

    fn read(&self, out: TensorHandle) -> Vec<f32> {
        let host = HostData::from_tensor_handle(&self.client, out, HostDataType::F32);
        host.iter_indexed_f32().map(|(_, value)| value).collect()
    }

    /// Times `routine` at `settings`: compiled and checked once, then the median of 7 rounds of
    /// 10 launches.
    fn time<M, SM, R>(&mut self, name: &str, routine: &R, settings: &R::Settings)
    where
        M: cubek_microkernel::attention::AttentionMicrokernel,
        R: AttentionRoutine<M, SM>,
        R::Settings: core::fmt::Display,
    {
        let label = format!("{name},{settings}");
        if std::env::var("CUBEK_LAB_PREPARE").is_ok() {
            let problem = self.tensors().problem();
            let start = Instant::now();
            for _ in 0..20000 {
                let _ = std::hint::black_box(routine.prepare(&self.hw, &problem, settings));
            }
            println!(
                "{label},prepare {:.2} us",
                start.elapsed().as_secs_f64() / 20000.0 * 1e6
            );
            let tensors_start = Instant::now();
            for _ in 0..20000 {
                let _ = std::hint::black_box(self.tensors());
            }
            println!(
                "{label},tensors {:.2} us",
                tensors_start.elapsed().as_secs_f64() / 20000.0 * 1e6
            );
            return;
        }
        let prepared = routine.prepare(&self.hw, &self.tensors().problem(), settings);
        let (cubes, smem, dim) = match &prepared {
            Ok((_, grid)) => (
                format!("{:?}", grid.launch.cube_count),
                grid.shared_memory,
                grid.launch.cube_dim.num_elems(),
            ),
            Err(why) => {
                println!("{label},declined,,,,,\"{why:?}\"");
                return;
            }
        };
        let run = catch_unwind(AssertUnwindSafe(|| {
            let out = routine
                .launch_on(&self.hw, self.tensors(), settings)
                .expect("prepared");
            let values = self.read(out);
            // Launches a round: enough that the first launch's host latency and the final sync
            // weigh little against the device's time (`CUBEK_LAB_LAUNCHES`, 10 by default).
            let launches: usize = std::env::var("CUBEK_LAB_LAUNCHES")
                .ok()
                .and_then(|n| n.parse().ok())
                .unwrap_or(10);
            let mut rounds = Vec::new();
            for _ in 0..7 {
                future::block_on(self.client.sync()).unwrap();
                let start = Instant::now();
                for _ in 0..launches {
                    let _ = routine.launch_on(&self.hw, self.tensors(), settings);
                }
                future::block_on(self.client.sync()).unwrap();
                rounds.push(start.elapsed().as_secs_f64() / launches as f64);
            }
            rounds.sort_by(f64::total_cmp);
            (values, rounds[3])
        }));
        match run {
            Ok((values, secs)) => {
                if self.reference.is_none() {
                    // A reference saved by an unmodified run, where one is named.
                    if let Ok(path) = std::env::var("CUBEK_LAB_REF")
                        && let Ok(bytes) = std::fs::read(&path)
                    {
                        self.reference = Some(
                            bytes
                                .chunks_exact(4)
                                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                                .collect(),
                        );
                    }
                }
                if let Ok(path) = std::env::var("CUBEK_LAB_SAVE_REF")
                    && self.reference.is_none()
                {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(path, bytes).expect("the reference is saved");
                }
                let diff = match &self.reference {
                    None => {
                        self.reference = Some(values);
                        0.0
                    }
                    Some(reference) => reference
                        .iter()
                        .zip(&values)
                        .map(|(a, b)| (a - b).abs())
                        .fold(
                            0.0f32,
                            |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) },
                        ),
                };
                println!(
                    "{label},{:.1},{:.1},\"{cubes}\",{dim},{smem},{diff:.4}",
                    secs * 1e6,
                    self.flops / secs / 1e12
                );
            }
            Err(_) => println!("{label},panicked,,,,,"),
        }
    }
}

fn count(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn fragment(rows: usize, keys: usize) -> CubeAttentionPlaneSettings {
    CubeAttentionPlaneSettings::Fragment(PlaneFragmentAttentionMicrokernelSettings::fragments(
        rows, keys,
    ))
}

const RINGS: [(&str, StageRing); 5] = [
    ("single", StageRing::SINGLE),
    ("double", StageRing::DOUBLE),
    ("triple", StageRing::TRIPLE),
    ("single_regs", StageRing::SINGLE_THROUGH_REGISTERS),
    ("double_regs", StageRing::DOUBLE_THROUGH_REGISTERS),
];

/// The cube microkernel settings of a suite: the production winner first, as the reference.
fn cube_settings(suite: &str) -> Vec<(String, CubeAttentionMicrokernelSettings)> {
    let winner = CubeAttentionMicrokernelSettings {
        plane: fragment(1, 8),
        heads: count(4),
        planes_along_queries: count(2),
        planes_along_keys: count(1),
        ring: StageRing::SINGLE,
        layout: StageLayout::SwizzledBlocks,
        delivery: lab_delivery(),
    };
    let mut out = vec![("winner".to_string(), winner)];
    let boxes: &[(usize, usize)] = match suite {
        "quick" => &[(1, 8)],
        _ => &[(1, 4), (1, 8), (2, 4), (2, 8), (1, 16)],
    };
    for &(rows, keys) in boxes {
        for heads in lab_list("CUBEK_LAB_HEADS", &[1, 2, 4]) {
            for planes in lab_list("CUBEK_LAB_PQ", &[1, 2, 4]) {
                for (ring_name, ring) in RINGS {
                    for (layout_name, layout) in [
                        ("swizzled", StageLayout::SwizzledBlocks),
                        ("padded_blocks", StageLayout::PaddedBlocks),
                    ] {
                        if suite == "quick" && layout_name != "swizzled" {
                            continue;
                        }
                        out.push((
                            format!("f{rows}x{keys} h{heads} pq{planes} {ring_name} {layout_name}"),
                            CubeAttentionMicrokernelSettings {
                                plane: fragment(rows, keys),
                                heads: count(heads),
                                planes_along_queries: count(planes),
                                planes_along_keys: count(1),
                                ring,
                                layout,
                                delivery: lab_delivery(),
                            },
                        ));
                    }
                }
            }
        }
    }
    out
}

/// A comma-separated list of counts in `var`, else `default`.
fn lab_list(var: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(var)
        .ok()
        .map(|list| {
            list.split(',')
                .filter_map(|n| n.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

/// Cubes a box the hyper suites try: `CUBEK_LAB_SPLITS`, comma separated, else 2, 3, 4, 6 and 8.
fn lab_splits() -> Vec<usize> {
    std::env::var("CUBEK_LAB_SPLITS")
        .ok()
        .map(|list| {
            list.split(',')
                .filter_map(|n| n.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![2, 3, 4, 6, 8])
}

/// The delivery the lab asks of every setting: `CUBEK_LAB_KV_DELIVERY=async` for the copy engine.
fn lab_delivery() -> CubeStagedDelivery {
    match std::env::var("CUBEK_LAB_KV_DELIVERY").as_deref() {
        Ok("async") => CubeStagedDelivery::AsyncCopy,
        _ => CubeStagedDelivery::Copy,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize| args[i].parse::<usize>().expect("a count");
    let (heads, kv_heads, queries, keys) = (num(1), num(2), num(3), num(4));
    let suite = args.get(5).map(String::as_str).unwrap_or("quick");
    let mut lab = Lab::new(heads, kv_heads, queries, keys);
    eprintln!(
        "lab: {heads}/{kv_heads} heads, {queries} queries over {keys} keys, {:.2} GFLOP, delivery {:?}",
        lab.flops / 1e9,
        std::env::var("CUBEK_LAB_KV_DELIVERY").unwrap_or_else(|_| "sync".into())
    );
    println!("routine,settings,us,tflops,cubes,cube_dim,smem,maxdiff");
    let settings = cube_settings(suite);
    let filter = std::env::var("CUBEK_LAB_FILTER").ok();
    for (name, microkernel) in &settings {
        if name == "winner" && std::env::var("CUBEK_LAB_NO_WINNER").is_ok() {
            continue;
        }
        if let Some(filter) = &filter
            && name != "winner"
            && !filter.split('|').any(|part| name.contains(part))
        {
            continue;
        }
        let cube = CubeStagedAttentionRoutineSettings {
            microkernel: *microkernel,
            // `CUBEK_LAB_LAST_FIRST`: the longest causal boxes launch first.
            queries: match std::env::var("CUBEK_LAB_LAST_FIRST") {
                Ok(_) => QueryOrder::LastFirst,
                Err(_) => QueryOrder::InOrder,
            },
        };
        lab.time(&format!("cube {name}"), &CubeStagedAttentionRoutine, &cube);
        if suite.contains("stream") {
            for per_worker in lab_list("CUBEK_LAB_PER_WORKER", &[1]) {
                let stream = HypercubeStreamFragmentAttentionRoutineSettings {
                    microkernel: HypercubeAttentionMicrokernelSettings { cube: *microkernel },
                    cubes_per_worker: count(per_worker),
                };
                lab.time(
                    &format!("stream{per_worker} {name}"),
                    &HypercubeStreamFragmentAttentionRoutine,
                    &stream,
                );
            }
        }
        if suite.contains("hyper") {
            for cubes in lab_splits() {
                let hyper = HypercubeSecondLaunchFragmentAttentionRoutineSettings {
                    microkernel: HypercubeAttentionMicrokernelSettings { cube: *microkernel },
                    cubes_along_keys: count(cubes),
                };
                lab.time(
                    &format!("hyper{cubes} {name}"),
                    &HypercubeSecondLaunchFragmentAttentionRoutine,
                    &hyper,
                );
            }
        }
    }
}
