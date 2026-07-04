//! Minimal driver for profiling the QR decomposition under nsys/ncu.
//!
//! Run with:
//! ```sh
//! cargo build --release -p cubek-linalg --example profile_qr --features cubecl/cuda
//! nsys profile --stats=true target/release/examples/profile_qr [m] [n] [iters]
//! ```

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, TestRuntime, future};

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2048);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(2048);
    let iters: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);

    let client = TestRuntime::client(&Default::default());

    // Col-major pseudo-random data, same layout the tests use.
    let mut state = 0x2545F4914F6CDD1Du64;
    let data: Vec<f32> = (0..m * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) as f32 - 0.5
        })
        .collect();
    let handle = client.create_from_slice(f32::as_bytes(&data));
    let a = TensorHandle::<TestRuntime>::new(
        handle,
        vec![m, n],
        vec![1, m],
        f32::as_type_native_unchecked(),
    );

    // Warmup (JIT compilation).
    cubek_linalg::qr::<TestRuntime, f32>(&client, &a).unwrap();
    future::block_on(client.sync()).unwrap();

    for i in 0..iters {
        let start = std::time::Instant::now();
        cubek_linalg::qr::<TestRuntime, f32>(&client, &a).unwrap();
        future::block_on(client.sync()).unwrap();
        println!("iter {i}: {:?}", start.elapsed());
    }
}
