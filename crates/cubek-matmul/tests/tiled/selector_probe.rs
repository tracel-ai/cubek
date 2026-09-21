//! A manual read on what [`CmmaRoutine`](cubek_matmul::tiled::cmma::CmmaRoutine)'s selector
//! picks, against the multi-level routine it ports. The plan it states is arithmetic over the
//! device's own numbers, and this is where that arithmetic meets a clock.

use cubek_matmul::{
    eval::benchmarks::gemm::{GemmProblem, bench, problems, strategies},
    strategy::Strategy,
};
use cubek_test_utils::CatalogEntry;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn median_ms(strategy: &Strategy, problem: &GemmProblem) -> Option<f64> {
    let samples = bench(strategy, problem, 10).ok()?;
    let mut durations = samples.durations.clone();
    durations.sort();
    Some(durations[durations.len() / 2].as_secs_f64() * 1e3)
}

/// The selector's own pick against multi-level, over every row-major f16 shape.
///
/// Run manually: `cargo test --release --features cubecl/cuda,benchmarks -p cubek-matmul
/// --test lib selector_vs_multi_level -- --ignored --nocapture`
#[test]
#[ignore = "timing probe, run manually"]
fn selector_vs_multi_level() {
    for entry in problems() {
        if !entry.id.ends_with("_rr_f16") {
            continue;
        }
        let problem = entry.value;
        let dsl = median_ms(&lookup(strategies(), "cmma"), &problem);
        let multi = median_ms(&lookup(strategies(), "simple_cyclic_cmma"), &problem);
        match (dsl, multi) {
            (Some(dsl), Some(multi)) => println!(
                "{:<34} dsl {dsl:8.3} ms   multi {multi:8.3} ms   x{:.2}",
                entry.id,
                multi / dsl
            ),
            // A shape one of the two declines: the cmma transport cannot mask an overhang, so
            // an `m` under one instruction row has no plan here.
            (dsl, multi) => println!("{:<34} dsl {dsl:?} multi {multi:?}", entry.id),
        }
    }
}
