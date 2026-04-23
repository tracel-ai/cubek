//! Minimal reproducer for a numerical regression introduced by
//! cubek PR #191 (commit 662610b7, "Refactor matmul types for a
//! Tile enum").
//!
//! What this does:
//!   1. Builds two deterministic NCHW tensors (no weights file, no
//!      images — everything synthesized from an LCG seed, then
//!      normalized with LPIPS's published mean/stdev).
//!   2. Runs the same LPIPS-shaped pipeline (VGG-16 feature blocks +
//!      per-block L2 normalization + 1x1 head convs + spatial mean)
//!      on the wgpu backend and on the ndarray CPU backend.
//!   3. Compares the final scalar and asserts `|wgpu - ndarray| < 1e-5`.
//!
//! Measured on an M-series Mac with burn @ 0e05dc6e:
//!   cubek be1fef47 (parent of #191): diff = 2.91e-10  PASS
//!   cubek 662610b7 (#191 itself)   : diff = 3.28e-05  FAIL
//!   cubek ac4b2e48 (post-#191)     : diff = 3.28e-05  FAIL
//!
//! Matches Brush's LPIPS VGG test regression on the same bisect points.
//!
//! Dropping the 1x1 head conv from each block — or going below ~128
//! channels — makes the drift vanish, which is consistent with the
//! regression living in the tiled-matmul path #191 refactored.
//!
//! Run with `cargo run --release`.

use burn::backend::NdArray;
type Wgpu = burn_wgpu::Wgpu<f32, i32>;
use burn::tensor::module::{conv2d, max_pool2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData, activation::relu, backend::Backend};

// Tiny LCG so we don't need a rand dep. Gives byte-identical data
// across runs, platforms, and backends.
fn lcg_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (s >> 40) as u32;
            (bits as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tensor_from_seed<B: Backend, const D: usize>(
    shape: [usize; D],
    seed: u64,
    scale: f32,
    device: &B::Device,
) -> Tensor<B, D> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = lcg_floats(n, seed).into_iter().map(|x| x * scale).collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn conv_layer<B: Backend>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
) -> Tensor<B, 4> {
    // 3x3 conv, stride 1, padding 1 — matches VGG's block shape.
    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
    conv2d(x, weight, Some(bias), opts)
}

fn head_1x1<B: Backend>(x: Tensor<B, 4>, weight: Tensor<B, 4>) -> Tensor<B, 4> {
    // 1x1 conv, no padding, no bias — matches LPIPS's learned heads.
    let opts = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    conv2d(x, weight, None, opts)
}

// Computes the LPIPS-style reduction of two inputs through the same
// shared conv stack: per block, normalize features per pixel, take
// the squared diff, apply a learned 1x1 head conv, and spatially
// average. Sum the per-block scalars.
async fn lpips_like<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
    device: &B::Device,
) -> f32 {
    let blocks: &[(usize, usize, usize)] = &[
        (2, 3, 64),
        (2, 64, 128),
        (3, 128, 256),
        (3, 256, 512),
        (3, 512, 512),
    ];

    let mut a = a;
    let mut b = b;
    let mut loss = 0f32;
    let mut seed: u64 = 0x100;
    for (b_idx, &(n_convs, in_c, out_c)) in blocks.iter().enumerate() {
        if b_idx != 0 {
            a = max_pool2d(a, [2, 2], [2, 2], [0, 0], [1, 1], false);
            b = max_pool2d(b, [2, 2], [2, 2], [0, 0], [1, 1], false);
        }
        for c_idx in 0..n_convs {
            let ic = if c_idx == 0 { in_c } else { out_c };
            let w = tensor_from_seed::<B, 4>([out_c, ic, 3, 3], seed, 0.1, device);
            let bias = tensor_from_seed::<B, 1>([out_c], seed ^ 0xBBBB, 0.01, device);
            seed = seed.wrapping_add(0x100);
            a = relu(conv_layer(a, w.clone(), bias.clone()));
            b = relu(conv_layer(b, w, bias));
        }

        // LPIPS-style per-pixel L2 normalization across channels.
        let norm_a = a.clone().powi_scalar(2).sum_dim(1).sqrt();
        let norm_b = b.clone().powi_scalar(2).sum_dim(1).sqrt();
        let na = a.clone() / (norm_a + 1e-10);
        let nb = b.clone() / (norm_b + 1e-10);
        let diff = (na - nb).powi_scalar(2);

        // 1x1 head conv reduces channels → 1. LPIPS has one per block.
        let head_w = tensor_from_seed::<B, 4>([1, out_c, 1, 1], seed ^ 0xDEAD, 0.05, device);
        seed = seed.wrapping_add(0x100);
        let class = head_1x1(diff, head_w);
        let block_loss: Tensor<B, 1> = class.mean_dim(2).mean_dim(3).reshape([1]);
        let scalar = block_loss
            .into_data_async()
            .await
            .expect("readback")
            .as_slice::<f32>()
            .expect("f32")[0];
        loss += scalar;
    }
    loss
}

fn make_inputs<B: Backend>(device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // LPIPS-matched: 128x128 input, NCHW, LPIPS mean/scale applied.
    let raw_a = tensor_from_seed::<B, 4>([1, 3, 128, 128], 0xAAAA_AAAA, 1.0, device);
    let raw_b = tensor_from_seed::<B, 4>([1, 3, 128, 128], 0xBBBB_BBBB, 1.0, device);
    let shift = Tensor::<B, 1>::from_data(
        TensorData::new(vec![-0.030f32, -0.088, -0.188], [3]),
        device,
    )
    .reshape([1, 3, 1, 1]);
    let scale = Tensor::<B, 1>::from_data(
        TensorData::new(vec![0.458f32, 0.448, 0.450], [3]),
        device,
    )
    .reshape([1, 3, 1, 1]);
    let a = (raw_a - shift.clone()) / scale.clone();
    let b = (raw_b - shift) / scale;
    (a, b)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let wgpu_device = burn_wgpu::WgpuDevice::DefaultDevice;
    let nd_device = Default::default();

    let (wa, wb) = make_inputs::<Wgpu>(&wgpu_device);
    let (na, nb) = make_inputs::<NdArray>(&nd_device);

    let wgpu_ab = lpips_like::<Wgpu>(wa, wb, &wgpu_device).await;
    let nd_ab = lpips_like::<NdArray>(na, nb, &nd_device).await;

    let diff = (wgpu_ab - nd_ab).abs();
    println!("LPIPS-like(a, b) on wgpu    = {wgpu_ab:.9}");
    println!("LPIPS-like(a, b) on ndarray = {nd_ab:.9}");
    println!("|wgpu - ndarray|             = {diff:.3e}");

    // Pre-cubek-#191 agrees with ndarray to ~1e-5 on this synthetic
    // LPIPS-shaped pipeline; post-#191 drifts well past that.
    const TOL: f32 = 1e-5;
    assert!(
        diff < TOL,
        "wgpu drifted from ndarray by {diff:.3e} (> {TOL:.0e}); \
         reproduces the cubek #191 matmul regression"
    );
    println!("OK: backends agree within {TOL:.0e}");
}
