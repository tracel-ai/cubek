//! Manual timing probes for the software register MMA leaf.
//!
//! Run one backend at a time so `TestRuntime` selects it:
//!
//! ```text
//! cargo test-cpu-benchmark -p cubek-tile register_leaf_benchmark -- --ignored --nocapture
//! cargo test-wgpu-benchmark -p cubek-tile register_leaf_benchmark -- --ignored --nocapture
//! ```
//!
//! The geometry is device-derived rather than fixed. A GPU needs two things a CPU does not, and
//! without them the whole problem lands on one thread and the numbers say nothing about the leaf:
//!
//! - **A full grid.** One cube per output block ([`Cut::cube`]), so the cube count scales with the
//!   problem instead of being pinned at one.
//! - **Work on the plane's lanes.** [`Cut::unit`] on `N`, so the block's columns are dealt across
//!   the plane rather than every lane recomputing the same cell. `plane_size` is `1` on CPU, which
//!   collapses this level to one lane per cube; the same space therefore serves both backends, and
//!   the problem extents stay fixed so `GFLOP/s` is comparable across them.
//!
//! Both callers of the leaf's shared `contract_block` are timed per case: the memory-backed leaf,
//! which seeds the register block from the output and commits it back on every `K` visit, and the
//! promoted block, which holds the accumulator across the whole walk. The gap between them is the
//! cost of that round trip.

use std::time::{Duration, Instant};

use cubecl::{Runtime, TestRuntime, client::ComputeClient, future, prelude::*};
use cubek_test_utils::TileInput;
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const SAMPLES: usize = 7;
const REPEATS: usize = 3;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    /// Rows of the output each cube owns. Its columns are the plane's business, so the block's
    /// `N` extent is derived from `plane_size` rather than stated here.
    block_m: usize,
    /// The register tile: `leaf.0 × leaf.1 / rhs_width` accumulator lines, `leaf.2` deep.
    leaf: (usize, usize, usize),
    /// Lhs line width along `K`. This is what the leaf's (line, lane) `K` walk exists for: at `1`
    /// there is one lane per line and the walk is the old flat one.
    lhs_width: usize,
    /// Rhs and output line width along `N`.
    rhs_width: usize,
}

const CASES: [Case; 5] = [
    Case {
        name: "block_32",
        m: 512,
        n: 2048,
        k: 256,
        block_m: 32,
        leaf: (8, 16, 8),
        lhs_width: 4,
        rhs_width: 4,
    },
    // 64 accumulator lines: exactly the shader unroll budget, so the block stays fully inlined
    // on every backend and the next size up would not.
    Case {
        name: "block_64",
        m: 512,
        n: 2048,
        k: 256,
        block_m: 64,
        leaf: (16, 16, 8),
        lhs_width: 4,
        rhs_width: 4,
    },
    // Four times the contraction at the same block, isolating the K walk from the prologue.
    Case {
        name: "deep_k",
        m: 512,
        n: 2048,
        k: 1024,
        block_m: 32,
        leaf: (8, 16, 8),
        lhs_width: 4,
        rhs_width: 4,
    },
    // `block_32` with a scalar lhs. The leaf's (line, lane) walk degenerates to one lane per line
    // here, so the delta against `block_32` answers whether lining the lhs along `K` pays at all:
    // a wide lhs load feeding `leaf.2 / lhs_width` fan-outs, against one scalar load per step.
    Case {
        name: "scalar_lhs",
        m: 512,
        n: 2048,
        k: 256,
        block_m: 32,
        leaf: (8, 16, 8),
        lhs_width: 1,
        rhs_width: 4,
    },
    // Extents divisible by neither the block nor the leaf tile, so the edge instances take the
    // masked reads and writes. The line widths still divide each innermost buffer extent, so
    // vectorized serving stays legal.
    Case {
        name: "masked_edges",
        m: 508,
        n: 2044,
        k: 256,
        block_m: 32,
        leaf: (8, 16, 8),
        lhs_width: 4,
        rhs_width: 4,
    },
];

/// Which caller of the leaf's shared walk a run drives.
#[derive(Clone, Copy)]
enum Variant {
    /// The accumulator lives in the output: seeded and committed once per `K` tile visit.
    Memory,
    /// The accumulator is promoted to its own register block for the whole walk.
    Promoted,
}

/// This deliberately selects `Leaf::Memory`: `Tile::mma` lowers it to
/// `instruction::mma::register::mma_register_memory`, not CMMA/manual MMA.
#[cube(launch)]
fn register_leaf_kernel<E: Numeric, LV: Size, V: Size>(
    a: &TileArg<'_, E, LV>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// The same contraction with the accumulator promoted out of the output for the whole `K` walk.
#[cube(launch)]
fn register_leaf_promoted_kernel<E: Numeric, LV: Size, V: Size>(
    a: &TileArg<'_, E, LV>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.promote::<E, _>(&a);
    acc.zero();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}

/// A cube per `block_m × (leaf_n · plane_size)` block of the output, its columns dealt across the
/// plane's lanes and its `K` walked sequentially.
///
/// The block's `N` extent is exactly what one plane covers, which is the constraint `cube_dim`
/// enforces: a `Unit` axis must partition exactly `plane_size` lanes, so the level's `N` grid has
/// to be `plane_size` tiles wide. On CPU `plane_size` is `1` and the block narrows to a single
/// leaf tile, which turns the same space into many small sequential cubes.
fn benchmark_space(case: Case, plane_size: usize) -> Space {
    let (leaf_m, leaf_n, leaf_k) = case.leaf;
    let block_n = leaf_n * plane_size;
    Tiling::new()
        .extents(&[(M, case.m), (N, case.n), (K, case.k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, case.block_m))
                .axis(N, Cut::cube(CubeAxis::Y, block_n))
                .axis(K, Cut::sequential(case.k))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(leaf_m))
                .axis(N, Cut::unit(leaf_n))
                .axis(K, Cut::sequential(leaf_k))
        })
        .build()
        .resolve_lanes(plane_size)
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

/// Time one (case, variant) pair and print its median launch.
fn run(client: &ComputeClient<TestRuntime>, case: Case, variant: Variant, plane_size: usize) {
    let space = benchmark_space(case, plane_size);
    let cube_count = space.cube_count();
    let cube_dim = space.cube_dim(client);

    let a = TileInput::builder(client, space.project(&[M, K]))
        .untiled()
        .uniform(0, -1.0, 1.0);
    let b = TileInput::builder(client, space.project(&[K, N]))
        .untiled()
        .uniform(1, -1.0, 1.0);
    let c = TileInput::builder(client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let dtype = f32::elem_type_native();

    let launch = || match variant {
        Variant::Memory => register_leaf_kernel::launch::<TestRuntime>(
            client,
            cube_count.clone(),
            cube_dim,
            case.lhs_width,
            case.rhs_width,
            a.arg(),
            b.arg(),
            c.arg(),
            space.clone(),
            dtype,
        ),
        Variant::Promoted => register_leaf_promoted_kernel::launch::<TestRuntime>(
            client,
            cube_count.clone(),
            cube_dim,
            case.lhs_width,
            case.rhs_width,
            a.arg(),
            b.arg(),
            c.arg(),
            space.clone(),
            dtype,
        ),
    };

    for _ in 0..2 {
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
    let label = match variant {
        Variant::Memory => "memory",
        Variant::Promoted => "promoted",
    };
    let cubes = match cube_count {
        CubeCount::Static(x, y, z) => x * y * z,
        _ => 0,
    };
    println!(
        "{:<14} {:<9} {:>4} cubes x {:>3} lanes  median {:>8.3} ms  {:>7.2} GFLOP/s",
        case.name,
        label,
        cubes,
        cube_dim.num_elems(),
        elapsed.as_secs_f64() * 1e3,
        flops / elapsed.as_secs_f64() / 1e9,
    );
}

#[test]
#[ignore = "manual performance probe; run with a specific test-*-benchmark backend alias"]
fn register_leaf_benchmark() {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    println!(
        "register leaf benchmark on {} (plane_size {plane_size}): \
         {SAMPLES} samples x {REPEATS} launches/sample",
        <TestRuntime as Runtime>::name(&client),
    );

    for case in CASES {
        for variant in [Variant::Memory, Variant::Promoted] {
            run(&client, case, variant, plane_size);
        }
    }
}
