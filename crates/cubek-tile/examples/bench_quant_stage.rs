//! Stage-depth probe for the packed-quant smem stage on the register leaf.
//!
//!   cargo run --release --example bench_quant_stage --features cubecl/metal
//!
//! Measured on Metal (M-series), `q8s` packed weight, `n = k = 4096`. At an *equal* `tk` the packed
//! and f32 stages tie (within ~5%): both stream the same packed bytes from gmem, and the unpack is
//! not on the reuse path (the microkernel caches the rhs lines before the `mr` loop). The packed
//! stage pays off only by being 4x smaller, which buys stage depth: the f32 stage hits the 32 KB
//! threadgroup ceiling at `tk = 64` (`64·128·4 B`), while packed reaches `tk = 128` (~17 KB). Depth
//! helps the `m = 1` decode path (~21 vs ~16.5 GFLOP/s best-config, ~20-25%) and does nothing for
//! `m = 8`, where `tk = 32` is already best.
//!
//! To re-run the A/B, temporarily force `pack_quant = false` in `Staging::new` (staging/fill.rs)
//! and alternate the two binaries in one session; the arms are within noise, so single runs mislead.

use std::time::Instant;

use cubecl::{TestRuntime, prelude::*};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{HostData, HostDataType, TileInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// `C = A · dequant(B)`, `B` the packed weight — the staged lowering picks the stage form.
#[cube(launch)]
fn staged_matmul_quant_rhs<I: Numeric, E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, I>,
    c: &StridedTileArg<'_, E>,
    #[define(I)] _b_dtype: StorageType,
    #[define(E)] _e_dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile_dequant::<E>();
    let mut c = c.tile();
    c.mma(&a, &b);
}

struct Case {
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
    tk: usize,
    bn: usize,
}

fn main() {
    let packed = std::env::var("CUBEK_NO_PACK_QUANT").is_err();
    let arm = if packed { "packed" } else { "f32   " };
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // `m` is the reuse factor: the cube stages one B tile and reuses it for all `m` rows, so if
    // dequant-at-read costs per use it should show up as `m` grows. `bn` is the scale block along
    // the packed axis: staged scales cost `pack/bn` of the staged values (4 = worst the scheme
    // allows, 128 = production).
    // Sweep the stage depth `tk`. The B stage costs `tk·tn` bytes packed vs `4·tk·tn` as f32, so
    // f32 hits the threadgroup-memory ceiling ~4x sooner. If depth buys anything, the packed arm is
    // the only one that can afford it — that, not a faster same-config kernel, is where the win
    // would come from.
    let cases = [
        Case { label: "gemv m=1  tk=32 ", m: 1, n: 4096, k: 4096, tk: 32, bn: 128 },
        Case { label: "gemv m=1  tk=64 ", m: 1, n: 4096, k: 4096, tk: 64, bn: 128 },
        Case { label: "gemv m=1  tk=128", m: 1, n: 4096, k: 4096, tk: 128, bn: 128 },
        Case { label: "gemv m=1  tk=256", m: 1, n: 4096, k: 4096, tk: 256, bn: 128 },
        Case { label: "gemm m=8  tk=32 ", m: 8, n: 4096, k: 4096, tk: 32, bn: 128 },
        Case { label: "gemm m=8  tk=64 ", m: 8, n: 4096, k: 4096, tk: 64, bn: 128 },
        Case { label: "gemm m=8  tk=128", m: 8, n: 4096, k: 4096, tk: 128, bn: 128 },
    ];

    println!("arm={arm}  (CUBEK_NO_PACK_QUANT unset => packed stage)");
    for case in &cases {
        match run(&client, case) {
            Ok((ms, gflops)) => {
                println!("  {arm}  {}  {ms:8.3} ms  {gflops:8.1} GFLOP/s", case.label)
            }
            Err(e) => println!("  {arm}  {}  skipped: {e}", case.label),
        }
    }
}

fn run(client: &ComputeClient<TestRuntime>, case: &Case) -> Result<(f64, f64), String> {
    let &Case { m, n, k, tk, bn, .. } = case;

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, bn as u8]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    let pack = scheme.num_quants();
    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        return Err(format!("device vectors cap at {max_width} < pack {pack}"));
    }

    // L0 stages one `m × tn × tk` cube tile; L1 spreads that tile's `N` across the plane's lanes,
    // one served line each, so the leaf is `mr = m`, `nr = 1` — unrolled for every `m` here (the
    // `mr·nr <= 64` cliff), keeping the unroll state constant across cases.
    let lanes = client.properties().hardware.plane_size_max as usize;
    let un = pack; // one served line per lane
    let tn = lanes * un;
    if !n.is_multiple_of(tn) || !k.is_multiple_of(tk) {
        return Err(format!("shape does not divide the {tn}x{tk} cube tile"));
    }
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axis(M, Cut::sequential(m))
                .axis(N, Cut::cube(CubeAxis::X, tn))
                .axis(K, Cut::sequential(tk))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(m))
                .axis(N, Cut::unit(un))
                .axis(K, Cut::sequential(tk))
        })
        .leaf(Leaf::Register);

    let a = TileInput::builder(client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(client, space.project(&[K, N]))
        .untiled()
        .packed(&scheme)
        .arange();
    let c = TileInput::builder(client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let b_dtype = u32::as_type_native_unchecked().storage_type();
    let e_dtype = f32::as_type_native_unchecked().storage_type();

    let launcher = space.launcher(client);
    let launch = || {
        staged_matmul_quant_rhs::launch::<TestRuntime>(
            client,
            launcher.cube_count(),
            launcher.cube_dim(),
            launcher.arg(a.handle().binding()).subspace(&[M, K]).build(),
            launcher
                .arg(b.tile.handle().binding())
                .subspace(&[K, N])
                .vectorize(pack)
                .quantized(b.scales_arg(), scheme)
                .build(),
            launcher
                .arg(c.handle().binding())
                .subspace(&[M, N])
                .vectorize(pack)
                .build(),
            b_dtype,
            e_dtype,
        );
    };

    // Warm up (kernel build + caches), then time a fixed batch of launches to one sync.
    for _ in 0..3 {
        launch();
    }
    cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))?;

    // A kernel that fails validation silently emits zeros, which would time as a fast win. Refuse
    // to report a number until the output proves the contraction actually ran.
    let got = HostData::from_tensor_handle(client, c.handle(), HostDataType::F32);
    let live = (0..m * n).filter(|&i| got.get_f32(&[i / n, i % n]) != 0.0).count();
    if live == 0 {
        return Err("output is all zeros (kernel did not run)".to_string());
    }

    let iters = 20;
    let start = Instant::now();
    for _ in 0..iters {
        launch();
    }
    cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))?;
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    Ok((elapsed * 1e3, flops / elapsed / 1e9))
}
