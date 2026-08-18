//! Small, manual timing probes for the software register MMA leaf.
//!
//! Run one backend at a time so `TestRuntime` selects it:
//!
//! ```text
//! cargo test-cpu-benchmark -p cubek-tile register_leaf_benchmark -- --ignored --nocapture
//! cargo test-wgpu-benchmark -p cubek-tile register_leaf_benchmark -- --ignored --nocapture
//! ```
//!
//! The largest case holds three 64 x 64 f32 matrices (48 KiB total) and
//! performs 1.05 MFLOP per launch.

use std::time::{Duration, Instant};

use cubecl::{Runtime, TestRuntime, client::ComputeClient, future, prelude::*};
use cubek_test_utils::TileInput;
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const SAMPLES: usize = 15;
const REPEATS: usize = 10;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    edge: usize,
}

// `edge = 8` makes the first two cases fully unrolled at the leaf (64 register
// cells); the masked case exercises guarded loads/stores; deep-k isolates the
// contraction loop while keeping the output block compact.
const CASES: [Case; 4] = [
    Case {
        name: "compact_8x8",
        m: 32,
        n: 32,
        k: 128,
        edge: 8,
    },
    Case {
        name: "register_limit_8x8",
        m: 64,
        n: 64,
        k: 128,
        edge: 8,
    },
    Case {
        name: "deep_k",
        m: 32,
        n: 32,
        k: 512,
        edge: 8,
    },
    Case {
        name: "masked_edges",
        m: 30,
        n: 30,
        k: 128,
        edge: 8,
    },
];

/// This deliberately selects `Leaf::Memory`: `Tile::mma` lowers it to
/// `instruction::mma::register::mma_register_memory`, not CMMA/manual MMA.
#[cube(launch)]
fn register_leaf_kernel<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

fn benchmark_space(case: Case) -> Space {
    let spatial = |axis| Distribution::Spatial {
        scope: ComputeScope::Cube(axis),
        spread: Spread::Contiguous,
        coverage: Coverage::Instances(1),
    };
    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(M, case.edge), (N, case.edge), (K, case.edge)]),
        ByAxis::new(&[
            (M, spatial(CubeAxis::X)),
            (N, spatial(CubeAxis::Y)),
            (K, Distribution::Sequential),
        ]),
    )
    .buffered(Buffering::SINGLE);
    Space::new(&[(M, case.m), (N, case.n), (K, case.k)]).with_partitioner(partitioner)
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

#[test]
#[ignore = "manual performance probe; run with a specific test-*-benchmark backend alias"]
fn register_leaf_benchmark() {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    println!(
        "register leaf benchmark on {}: {SAMPLES} samples x {REPEATS} launches/sample",
        <TestRuntime as Runtime>::name(&client),
    );

    for case in CASES {
        let space = benchmark_space(case);
        let a = TileInput::builder(&client, space.project(&[M, K]))
            .untiled()
            .uniform(0, -1.0, 1.0);
        let b = TileInput::builder(&client, space.project(&[K, N]))
            .untiled()
            .uniform(1, -1.0, 1.0);
        let c = TileInput::builder(&client, space.project(&[M, N]))
            .untiled()
            .zeros();
        let dtype = f32::elem_type_native();

        let launch = || {
            register_leaf_kernel::launch::<TestRuntime>(
                &client,
                space.cube_count(),
                space.cube_dim(&client),
                a.arg(),
                b.arg(),
                c.arg(),
                space.clone(),
                dtype,
            );
        };
        for _ in 0..3 {
            launch();
        }
        future::block_on(client.sync()).expect("register-leaf warmup must complete");

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            for _ in 0..REPEATS {
                launch();
            }
            future::block_on(client.sync()).expect("register-leaf benchmark launch must complete");
            samples.push(started.elapsed() / REPEATS as u32);
        }

        let elapsed = median(samples);
        let flops = 2.0 * (case.m * case.n * case.k) as f64;
        let gflops = flops / elapsed.as_secs_f64() / 1e9;
        println!(
            "{:<22} {:>3}x{:<3}x{:<3}  median {:>8.3} ms  {:>7.2} GFLOP/s",
            case.name,
            case.m,
            case.n,
            case.k,
            elapsed.as_secs_f64() * 1e3,
            gflops,
        );
    }
}
