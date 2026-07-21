//! Minimal driver for profiling the QR decomposition under nsys/ncu.
//!
//! Run with:
//! ```sh
//! cargo build --release -p cubek-linalg --example profile_qr --features cubecl/cuda
//! nsys profile --stats=true target/release/examples/profile_qr [m] [n] [iters] [dtype]
//! ```
//!
//! `dtype` is `f32` (default) or `f64`; pass a 5th arg `tf32` to opt the
//! trailing-update GEMMs into tensor cores (see `BahtTsqrStrategy`).

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, TestRuntime, future};

fn run<E: Float + CubeElement>(m: usize, n: usize, iters: usize, label: &str, allow_tf32: bool) {
    let client = TestRuntime::client(&Default::default());

    // Col-major pseudo-random data, same layout the tests use.
    let mut state = 0x2545F4914F6CDD1Du64;
    let data: Vec<E> = (0..m * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state as f64 / u64::MAX as f64) - 0.5;
            <E as num_traits::NumCast>::from(v).unwrap()
        })
        .collect();
    let handle = client.create_from_slice(E::as_bytes(&data));
    let a = TensorHandle::<TestRuntime>::new(
        handle,
        vec![m, n],
        vec![1, m],
        E::as_type_native_unchecked(),
    );

    let strategy = cubek_linalg::routines::BahtTsqrStrategy { allow_tf32 };

    // Warmup (JIT compilation).
    cubek_linalg::qr_with_strategy::<TestRuntime, E>(&client, &a, strategy).unwrap();
    future::block_on(client.sync()).unwrap();

    let mut times_ms = Vec::with_capacity(iters);
    for i in 0..iters {
        let start = std::time::Instant::now();
        cubek_linalg::qr_with_strategy::<TestRuntime, E>(&client, &a, strategy).unwrap();
        future::block_on(client.sync()).unwrap();
        let elapsed = start.elapsed();
        times_ms.push(elapsed.as_secs_f64() * 1e3);
        println!("iter {i}: {elapsed:?}");
    }

    times_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mean: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let median = times_ms[times_ms.len() / 2];
    println!("{m}x{n} {label}: mean {mean:.3} ms, median {median:.3} ms");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2048);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2048);
    let iters: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let dtype = args.next().unwrap_or_else(|| "f32".to_string());
    // Optional 5th arg: `tf32` opts the trailing-update GEMMs into tensor cores.
    let allow_tf32 = args.next().as_deref() == Some("tf32");
    let label = if allow_tf32 { "tf32" } else { "full" };

    match dtype.as_str() {
        "f64" => run::<f64>(m, n, iters, &format!("f64/{label}"), allow_tf32),
        "f32" => run::<f32>(m, n, iters, &format!("f32/{label}"), allow_tf32),
        other => panic!("unsupported dtype `{other}`; use f32 or f64"),
    }
}
