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

/// What strip width is worth, at the shapes big enough for a band to miss the cache.
#[test]
#[ignore = "timing probe, run manually"]
fn swizzle_width_probe() {
    use cubek_matmul::{
        routine::BlueprintStrategy,
        tiled::{
            Strategy as Tiled,
            cmma::{CmmaBlueprint, CmmaDelivery, Partition},
            cpu_gemm::{InstructionShape, PlaneGrid},
        },
    };
    use cubek_tile::CubeOrder;

    for id in [
        "square_1x8192_rr_f16",
        "square_1x6144_rr_f16",
        "square_2x4096_rr_f16",
    ] {
        let problem: GemmProblem = match problems().into_iter().find(|e| e.id == id) {
            Some(e) => e.value,
            None => continue,
        };
        println!("\n## {id}");
        for order in [
            CubeOrder::RowMajor,
            CubeOrder::SwizzleRow(2),
            CubeOrder::SwizzleRow(4),
            CubeOrder::SwizzleRow(8),
            CubeOrder::SwizzleRow(16),
            CubeOrder::SwizzleCol(4),
            CubeOrder::SwizzleCol(8),
        ] {
            let strategy = Tiled::Cmma(BlueprintStrategy::Forced(CmmaBlueprint {
                instruction: InstructionShape {
                    m: 16,
                    n: 16,
                    k: 16,
                },
                partition: Partition { m: 4, n: 4 },
                planes: PlaneGrid { m: 4, n: 2 },
                stage_k: 64,
                buffering: 2,
                delivery: CmmaDelivery::Copy,
                order,
            }))
            .into();
            match median_ms(&strategy, &problem) {
                Some(t) => println!("  {order:?} : {t:.3} ms"),
                None => println!("  {order:?} : declined"),
            }
        }
    }
}
