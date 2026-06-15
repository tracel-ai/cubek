//! Quick speed comparison for the tiled dequantize kernel: **sequential** (one cube
//! walks every tile) vs **spatial** (one cube per tile, run in parallel). Ignored by
//! default — run explicitly:
//!
//!   cargo test -p cubek-quant --test tiled_bench -- --ignored --nocapture
//!
//! Each cube runs one thread (`CubeDim::new_single()`); the parallelism is purely the
//! number of cubes (`cube_count`). Bump `TENSOR`/`TILE_EDGE` for a bigger problem, but
//! keep the sequential case modest: a single GPU thread over a huge tensor can run for
//! seconds and trip the OS GPU watchdog (TDR).

use std::time::Instant;

use cubecl::{CubeDim, Runtime, TestRuntime, client::ComputeClient, future, prelude::*};
use cubek_test_utils::TileInput;
use cubek_tile::{
    Axis, ByAxis, ComputeScope, Coverage, CubeAxis, Distribution, Partitioner, Space, Spread,
    TileArgLaunch,
};

const SEED: u64 = 0xC0FF_EE00;

#[derive(Clone, Copy)]
enum Mode {
    Sequential,
    Spatial,
}

/// Row-major partitioner: every axis `Sequential` (→ one cube total) or `Spatial`
/// (→ one tile per cube). Rank ≤ 3.
fn partitioner(mode: Mode, axes: &[Axis], edges: &[usize]) -> Partitioner {
    const CUBE_AXES: [CubeAxis; 3] = [CubeAxis::X, CubeAxis::Y, CubeAxis::Z];
    let edges: Vec<(Axis, usize)> = axes.iter().copied().zip(edges.iter().copied()).collect();
    let dists: Vec<(Axis, Distribution)> = axes
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let dist = match mode {
                Mode::Sequential => Distribution::Sequential,
                Mode::Spatial => Distribution::Spatial {
                    scope: ComputeScope::Cube(CUBE_AXES[i]),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                },
            };
            (a, dist)
        })
        .collect();
    Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists)).direct()
}

fn space(mode: Mode, axes: &[Axis], extents: &[usize], edges: &[usize]) -> Space {
    let entries: Vec<(Axis, usize)> = axes.iter().copied().zip(extents.iter().copied()).collect();
    Space::new(&entries).with_partitioner(partitioner(mode, axes, edges))
}

/// Average seconds per launch for `mode` on `tensor` tiled by `tile_edge` (per-tensor
/// scale, which keeps setup trivial — the kernel cost is the same shape of work).
fn time_mode(
    client: &ComputeClient<TestRuntime>,
    mode: Mode,
    tensor: &[usize],
    tile_edge: usize,
    iters: usize,
) -> f64 {
    let rank = tensor.len();
    let ax: Vec<Axis> = (0..rank as u8).map(Axis).collect();
    let edges = vec![tile_edge; rank];
    let ones = vec![1usize; rank];

    let val_space = space(mode, &ax, tensor, &edges);
    let scl_space = space(mode, &ax, &ones, &ones); // per-tensor scale: 1×…×1 grid

    let values = TileInput::builder(client, val_space.clone())
        .untiled()
        .arange();
    let scales = TileInput::builder(client, scl_space)
        .untiled()
        .uniform(SEED, 0.25, 2.0);
    let output = TileInput::builder(client, val_space.clone())
        .untiled()
        .zeros();

    let cube_count = val_space.partitioner().cube_count(&val_space);
    let dtype = f32::as_type_native_unchecked().storage_type();

    let launch = || {
        cubek_quant::dequantize_tiled::dequantize::launch::<TestRuntime>(
            client,
            cube_count.clone(),
            CubeDim::new_single(),
            TileArgLaunch::new(values.tensor_arg(1), values.space(), values.storage()),
            TileArgLaunch::new(scales.tensor_arg(1), scales.space(), scales.storage()),
            TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
            dtype,
            dtype,
            dtype,
            1usize,
            1usize,
            1usize,
        );
    };

    // Warm up (kernel compilation), then time `iters` launches against one sync.
    launch();
    future::block_on(client.sync()).unwrap();

    let start = Instant::now();
    for _ in 0..iters {
        launch();
    }
    future::block_on(client.sync()).unwrap();
    start.elapsed().as_secs_f64() / iters as f64
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_sequential_vs_spatial() {
    let client = TestRuntime::client(&Default::default());
    let tensor = [1024usize, 1024];
    let tile_edge = 32;
    let iters = 10;

    let seq = time_mode(&client, Mode::Sequential, &tensor, tile_edge, iters);
    let par = time_mode(&client, Mode::Spatial, &tensor, tile_edge, iters);

    let tiles: usize = tensor.iter().map(|&d| d / tile_edge).product();
    println!("\ntensor {tensor:?}, tile_edge {tile_edge} ({tiles} tiles), {iters} iters/sample");
    println!("  sequential (1 cube):    {:>9.3} ms", seq * 1e3);
    println!("  spatial ({tiles} cubes):    {:>9.3} ms", par * 1e3);
    println!("  speedup:                {:>9.1}x", seq / par);
}
