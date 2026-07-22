//! Correctness over the matmul benchmark catalogues (gemm + gemv).

#![cfg(feature = "benchmarks")]

use cubecl::Runtime;
use cubek_matmul::eval::benchmarks::gemm::{GemmCorrectness, GemmProblem};
use cubek_matmul::eval::benchmarks::gemv::{GemvCorrectness, GemvProblem};
use cubek_matmul::strategy::Strategy;
use cubek_test_utils::{
    CatalogEntry, Correctness, TestOutcome, assert_equals_approx, skip_unless_cpu,
};

const SEEDS: [u64; 2] = [12, 34];

/// Loose epsilon. f16 matmul reductions accumulate noise quickly; tighten if
/// you ever need this catalogue to gate on precision regressions.
const F16_EPS: f32 = 0.5;

fn lookup<T>(entries: Vec<CatalogEntry<T>>, id: &str) -> T {
    entries
        .into_iter()
        .find(|e| e.id == id)
        .unwrap_or_else(|| panic!("unknown id: {id}"))
        .value
}

fn run_gemm(strategy_id: &str, problem_id: &str) {
    use cubek_matmul::eval::benchmarks::gemm::{problems, strategies};

    let strategy: Strategy = lookup(strategies(), strategy_id);
    let problem: GemmProblem = lookup(problems(), problem_id);

    let actual = match GemmCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = GemmCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

fn run_gemv(strategy_id: &str, problem_id: &str) {
    use cubek_matmul::eval::benchmarks::gemv::{problems, strategies};

    let strategy: Strategy = lookup(strategies(), strategy_id);
    let problem: GemvProblem = lookup(problems(), problem_id);

    let actual = match GemvCorrectness.kernel_result(&strategy, &problem, &SEEDS) {
        Ok(host) => host,
        Err(e) => return TestOutcome::CompileError(e).enforce(),
    };
    let expected = GemvCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap_or_else(|e| panic!("reference failed for {problem_id}: {e}"));

    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

#[test]
fn gemm_rect_1x512x512x512_rr_f16() {
    run_gemm("simple_cyclic_cmma", "rect_1x512x512x512_rr_f16");
}

#[test]
#[ignore = "slow CPU reference + CMMA fallbacks"]
fn gemm_square_2x1024_rr_f16() {
    run_gemm("simple_cyclic_cmma", "square_2x1024_rr_f16");
}

#[test]
#[ignore = "very slow CPU reference"]
fn gemm_square_1x6144_rr_f16() {
    run_gemm("simple_cyclic_cmma", "square_1x6144_rr_f16");
}

/// CpuGemm through the benchmark-catalog path (vs the extended tests' `test_matmul_strategy`
/// path). `vecmat` keeps the CPU reference cheap (`m = 1`).
#[test]
fn gemm_cpu_gemm_vecmat_2x1x4096x4096_rr_f32() {
    let client = cubecl::TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
    run_gemm("cpu_gemm", "vecmat_2x1x4096x4096_rr_f32");
}

#[test]
fn gemv_vecmat_b2_out4096_k8192_rr() {
    run_gemv("simple_vecmat", "vecmat_b2_out4096_k8192_rr");
}

#[test]
fn gemv_matvec_b2_out4096_k8192_rr() {
    run_gemv("simple_vecmat", "matvec_b2_out4096_k8192_rr");
}

/// Timing probe: the tile-DSL cyclic cmma vs the legacy SimpleAlgorithm it ports.
/// Run manually: `cargo test-metal-benchmark gemm_cmma_timing -- --ignored --nocapture`
#[test]
#[ignore = "timing probe, run manually"]
fn gemm_cmma_timing_vs_legacy() {
    use cubek_matmul::eval::benchmarks::gemm::{bench, problems, strategies};

    let problem: GemmProblem = lookup(problems(), "square_2x4096_rr_f16");
    for id in ["cmma", "simple_cyclic_cmma"] {
        let strategy: Strategy = lookup(strategies(), id);
        let samples = bench(&strategy, &problem, 10).unwrap();
        let mut ds = samples.durations.clone();
        ds.sort();
        println!(
            "{id}: median {:?} over {} samples, {:.2} TFLOPS",
            ds[ds.len() / 2],
            ds.len(),
            samples.tflops.unwrap_or(0.0)
        );
    }
}

/// Correctness at the forced legacy-shaped point (the crosspoint probe's blueprint):
/// the timing probe never validates, so this guards it.
#[test]
#[ignore = "crosspoint probe guard, run manually"]
fn gemm_cyclic_cmma_forced_point_correctness() {
    use cubek_matmul::eval::benchmarks::gemm::{GemmCorrectness, problems};
    use cubek_matmul::routines::BlueprintStrategy;
    use cubek_matmul::routines::cmma::{CmmaBlueprint, Partition};
    use cubek_matmul::routines::cpu_gemm::{Instruction, PlaneGrid};
    use cubek_test_utils::Correctness;

    let problem: GemmProblem = lookup(problems(), "rect_1x512x512x512_rr_f16");
    let forced = Strategy::Cmma(BlueprintStrategy::Forced(CmmaBlueprint {
        instruction: Instruction { m: 8, n: 8, k: 8 },
        partition: Partition { m: 1, n: 4 },
        planes: PlaneGrid { m: 4, n: 1 },
        stage_k: 32,
        delivery: cubek_tile::Delivery::Strided,
    }));
    let actual = GemmCorrectness
        .kernel_result(&forced, &problem, &SEEDS)
        .unwrap();
    let expected = GemmCorrectness
        .reference_result(&problem, &SEEDS, None)
        .unwrap();
    assert_equals_approx(&actual, &expected, F16_EPS)
        .as_test_outcome()
        .enforce();
}

/// Timing probe: each engine forced to the OTHER's working point, to split the gap
/// between tiling selection and codegen quality.
/// Run manually: `cargo test-metal-benchmark gemm_cyclic_cmma_crosspoint -- --ignored --nocapture`
#[test]
#[ignore = "timing probe, run manually"]
fn gemm_cyclic_cmma_crosspoint_timing() {
    use cubecl::ir::AddressType;
    use cubek_matmul::components::stage::PartitionBuffering;
    use cubek_matmul::components::tile::TileMatmulKind;
    use cubek_matmul::definition::{MatmulGlobalElems, MatmulProblem, TilingScheme};
    use cubek_matmul::eval::benchmarks::gemm::{bench, problems};
    use cubek_matmul::routines::BlueprintStrategy;
    use cubek_matmul::routines::cmma::{CmmaBlueprint, Partition};
    use cubek_matmul::routines::cpu_gemm::{Instruction, PlaneGrid};
    use cubek_std::MatrixLayout;

    let problem: GemmProblem = lookup(problems(), "square_2x4096_rr_f16");

    // The tile DSL forced to the legacy selector's point: 8x8x8 instruction, each plane
    // 1x4 tiles, 4x1 planes (128 units), stage 32x32, stage_k 32.
    let dsl_at_legacy_point = Strategy::Cmma(BlueprintStrategy::Forced(CmmaBlueprint {
        instruction: Instruction { m: 8, n: 8, k: 8 },
        partition: Partition { m: 1, n: 4 },
        planes: PlaneGrid { m: 4, n: 1 },
        stage_k: 32,
        delivery: cubek_tile::Delivery::Strided,
    }));

    // The legacy engine forced to the DSL selector's point: partition 2x8x4 per plane,
    // 4x2 planes (256 units), stage 64x128, stage_k 32.
    let f16 =
        cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16));
    let matmul_problem = MatmulProblem::from_parameters(
        4096,
        4096,
        4096,
        vec![2].into(),
        vec![2].into(),
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulGlobalElems {
            lhs: f16,
            rhs: f16,
            out: f16,
        },
        AddressType::U32,
    );
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((8, 8, 8).into())
        .with_partition_size((2, 8, 4).into())
        .with_stage_size((4, 2, 1).into())
        .build()
        .unwrap();
    let legacy_at_dsl_point = Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(
        cubek_matmul::definition::BatchMatmulBlueprint::builder(
            TileMatmulKind::Cmma,
            tiling_scheme,
            32,
            &matmul_problem,
        )
        .partition_buffering(PartitionBuffering::Single)
        .build(),
    ));

    // Legacy at its own tiling but WITHOUT the swizzled cube order (builder default),
    // to isolate how much of legacy's edge is the SwizzleRow(4) dispatch order.
    let legacy_tiling = TilingScheme::builder()
        .with_tile_size((8, 8, 8).into())
        .with_partition_size((1, 4, 4).into())
        .with_stage_size((4, 1, 1).into())
        .build()
        .unwrap();
    let legacy_no_swizzle = Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(
        cubek_matmul::definition::BatchMatmulBlueprint::builder(
            TileMatmulKind::Cmma,
            legacy_tiling,
            32,
            &matmul_problem,
        )
        .partition_buffering(PartitionBuffering::Single)
        .build(),
    ));

    // Thin tiling with deeper stages: the DSL fill phase is stage-count-bound, not
    // byte-bound, so fewer/deeper stages should shrink it at unchanged compute.
    let thin_deep = |stage_k: usize| {
        Strategy::Cmma(BlueprintStrategy::Forced(CmmaBlueprint {
            instruction: Instruction { m: 8, n: 8, k: 8 },
            partition: Partition { m: 1, n: 4 },
            planes: PlaneGrid { m: 4, n: 1 },
            stage_k,
            delivery: cubek_tile::Delivery::Strided,
        }))
    };

    use cubek_matmul::eval::benchmarks::gemm::strategies;
    for (id, strategy) in [
        ("dsl_at_own_point", lookup(strategies(), "cmma")),
        (
            "legacy_at_own_point",
            lookup(strategies(), "simple_cyclic_cmma"),
        ),
        ("dsl_at_legacy_point", dsl_at_legacy_point),
        ("legacy_at_dsl_point", legacy_at_dsl_point),
        ("legacy_own_no_swizzle", legacy_no_swizzle),
        ("dsl_thin_sk64", thin_deep(64)),
        ("dsl_thin_sk128", thin_deep(128)),
        ("dsl_thin_sk256", thin_deep(256)),
    ] {
        match bench(&strategy, &problem, 10) {
            Ok(samples) => {
                let mut ds = samples.durations.clone();
                ds.sort();
                println!(
                    "{id}: median {:?} over {} samples, {:.2} TFLOPS",
                    ds[ds.len() / 2],
                    ds.len(),
                    samples.tflops.unwrap_or(0.0)
                );
            }
            Err(e) => println!("{id}: setup error: {e}"),
        }
    }
}

/// Timing sweep: the tile-DSL cyclic cmma vs legacy across the row-major catalog, each
/// shape cross-checked (both engines must agree) so a broken kernel can't post a time.
/// Run manually: `cargo test-metal-benchmark gemm_cyclic_cmma_sweep -- --ignored --nocapture`
#[test]
#[ignore = "timing probe, run manually"]
fn gemm_cyclic_cmma_sweep() {
    use cubek_matmul::eval::benchmarks::gemm::{GemmCorrectness, bench, problems, strategies};
    use cubek_test_utils::Correctness;

    // Only shapes whose tensors stay under 64M elements: the harness's init kernel
    // dispatches a flattened grid and wgpu rejects >= 65536 groups (device-thread
    // panic), which rules out square_1x8192/16x2048/1024x512 and the b=4096 skinnies.
    let shapes = [
        "square_2x4096_rr_f16",
        "square_1x6144_rr_f16",
        "square_2x1024_rr_f16",
        "square_1x1536_rr_f16",
        "rect_1x512x512x512_rr_f16",
        "square_2x4096_rr_f32",
        "square_2x1024_rr_f32",
        "square_1x1536_rr_f32",
    ];
    let dsl: Strategy = lookup(strategies(), "cmma");
    let legacy: Strategy = lookup(strategies(), "simple_cyclic_cmma");

    for tag in shapes {
        let problem: GemmProblem = lookup(problems(), tag);

        let dsl_out = match GemmCorrectness.kernel_result(&dsl, &problem, &SEEDS) {
            Ok(out) => out,
            Err(e) => {
                println!("{tag}: dsl setup error: {e}");
                continue;
            }
        };

        let t = |s: &Strategy| {
            let samples = bench(s, &problem, 10).unwrap();
            let mut ds = samples.durations.clone();
            ds.sort();
            (ds[ds.len() / 2], samples.tflops.unwrap_or(0.0))
        };

        // Legacy flattens its cube count and wgpu rejects >= 65536 cubes (device-thread
        // panic, uncatchable), so those shapes report the DSL alone. Its stage is 32x32
        // at every catalog shape (8x8x8 instruction on Metal).
        let legacy_cubes = problem.m.div_ceil(32) * problem.n.div_ceil(32) * problem.b;
        if legacy_cubes >= 65536 {
            let (dt, dtf) = t(&dsl);
            println!("{tag}: dsl {dtf:.2} TFLOPS ({dt:?}); legacy exceeds its dispatch limit");
            continue;
        }

        let legacy_out = GemmCorrectness
            .kernel_result(&legacy, &problem, &SEEDS)
            .unwrap();
        // The DSL accumulates in f16 where legacy upgrades to f32, so the drift between
        // them grows ~sqrt(k); scale the tolerance accordingly (0.5 at k = 512).
        let eps = F16_EPS * (problem.k as f32 / 512.0).sqrt().max(1.0) * 2.0;
        if let cubek_test_utils::ValidationResult::Fail(e) =
            assert_equals_approx(&dsl_out, &legacy_out, eps)
        {
            println!("{tag}: MISMATCH vs legacy: {e}");
            continue;
        }

        let (dt, dtf) = t(&dsl);
        let (lt, ltf) = t(&legacy);
        println!(
            "{tag}: dsl {dtf:.2} TFLOPS ({dt:?}) vs legacy {ltf:.2} ({lt:?}) = {:.0}%",
            dtf / ltf * 100.0
        );
    }
}

/// Print the backend's cmma configs (debugging aid).
#[test]
#[ignore = "debug probe"]
fn print_cmma_configs() {
    let client = cubecl::TestRuntime::client(&Default::default());
    for c in client.properties().features.matmul.cmma.iter() {
        println!("{:?}", c);
    }
}
