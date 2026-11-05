//! A measurement, not a check: the cube shuffle's tensor-core planes against an NVFP4 weight at a
//! decode's few rows, their rows along `m` beside their rows along `n`, so it can be decided where
//! the rows-along-`n` planes are worth listing. Ignored; run it on a device offering the 16-bit
//! `16 × 8 × 16` mma, on mains power:
//!
//! ```text
//! cargo test -p cubek-matmul --features cubecl/cuda --test lib -- rows_along_n_bench --ignored --nocapture
//! ```

use super::indexed::upload;
use super::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{CubeShuffleMatmulRoutine, MatmulRoutine, MatmulTensors};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::cube::CubeShufflePlaneSettings;
use cubek_microkernel::matmul::plane::PlaneFragmentRowsAlong;
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::client;
use half::f16;
use std::time::Instant;

/// Qwen3-8B's decode projections, `(name, n, k)`.
const PROJECTIONS: [(&str, usize, usize); 4] = [
    ("qkv", 6144, 4096),
    ("o", 4096, 4096),
    ("gate", 12288, 4096),
    ("down", 4096, 12288),
];

/// The rows timed: one, a fragment's eight along `n`, and past it up to sixteen.
const ROWS: [usize; 6] = [1, 4, 8, 9, 12, 16];

/// Launches timed back to back in one round, and the rounds whose fastest is kept.
const LAUNCHES: usize = 50;
const ROUNDS: usize = 5;

#[test]
#[ignore = "a measurement: run by hand on a tensor-core device"]
fn rows_along_n_bench() {
    let client = client();
    let hw = Hardware::from_client(&client);
    let routine = CubeShuffleMatmulRoutine;
    let settings = routine.packed_settings(&hw);
    println!("setting | projection | rows | us");
    let mut summary = Vec::new();
    for (name, n, k) in PROJECTIONS {
        let weight = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format: PackedFormat::Nvfp4,
                operand: MatmulOperand::Rhs,
                slabs: 1,
                rows: n,
                k,
                seed: 2,
            },
        );
        for m in ROWS {
            let activation: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
                .collect();
            let lhs = upload(&client, &[m, k], &activation);
            let mut best = [None::<f64>; 3];
            for setting in &settings {
                let class = match setting.microkernel.plane {
                    CubeShufflePlaneSettings::Shuffle(_) => 0,
                    CubeShufflePlaneSettings::Fragment(plane) => match plane.rows_along {
                        PlaneFragmentRowsAlong::M => 1,
                        PlaneFragmentRowsAlong::N => 2,
                    },
                };
                let Some(us) = time(&client, &hw, &routine, setting, &lhs, &weight) else {
                    continue;
                };
                println!("{setting} | {name} | {m} | {us:.1}");
                if best[class].is_none_or(|fastest| us < fastest) {
                    best[class] = Some(us);
                }
            }
            summary.push((name, m, best));
        }
    }
    println!();
    println!("projection | rows | software | rows along m | rows along n | n vs m");
    for (name, m, best) in summary {
        let cell = |at: Option<f64>| match at {
            Some(us) => format!("{us:.1}"),
            None => "-".into(),
        };
        let ratio = match (best[1], best[2]) {
            (Some(along_m), Some(along_n)) => format!("{:.2}x", along_n / along_m),
            _ => "-".into(),
        };
        println!(
            "{name} | {m} | {} | {} | {} | {ratio}",
            cell(best[0]),
            cell(best[1]),
            cell(best[2])
        );
    }
}

/// Microseconds a launch of `setting` takes, the fastest round's average, or `None` where the
/// setting does not prepare the product.
fn time(
    client: &Client,
    hw: &Hardware,
    routine: &CubeShuffleMatmulRoutine,
    setting: &<CubeShuffleMatmulRoutine as Routine>::Settings,
    lhs: &PlainTensor,
    weight: &PackedSlabs,
) -> Option<f64> {
    let tensors = MatmulTensors::written_in(
        client,
        StoredTensor::Plain(lhs.clone()),
        StoredTensor::Packed(weight.matrix()),
        f16::elem_type_native(),
    )
    .ok()?;
    let (blueprint, grid) = routine.prepare(hw, &tensors.problem(), setting).ok()?;
    let launch = || routine.dispatch_into_buffer(&tensors, blueprint.clone(), grid.clone());
    for _ in 0..LAUNCHES {
        launch();
    }
    cubecl::future::block_on(client.sync()).expect("the warm-up completes");
    (0..ROUNDS)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..LAUNCHES {
                launch();
            }
            cubecl::future::block_on(client.sync()).expect("the round completes");
            start.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64
        })
        .min_by(f64::total_cmp)
}
