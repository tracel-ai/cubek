//! Correctness path for the GEMM category. Mirrors `benchmark.rs` but only
//! runs the kernel once, captures the output as `HostData`, and either:
//!
//! - compares against the in-process CPU reference (`Cpu` mode), or
//! - writes the output to a file (`WriteReference`), or
//! - reads a previously-written file and compares against it (`ValidateReference`).
//!
//! Inputs are seeded deterministically off `(strategy_id, problem_id)` so two
//! commits at the same workspace path produce the same input bits. The actual
//! launch + comparison plumbing lives in `cubek_matmul::cpu_reference` so the
//! GEMM tests and the tuner share one code path.

use cubecl::{
    Runtime, TestRuntime,
    client::ComputeClient,
    ir::AddressType,
    zspace::{Shape, shape},
};
use cubek::{
    matmul::{
        cpu_reference::{CorrectnessReport, CorrectnessTarget, validate_strategy},
        definition::{MatmulElems, MatmulPrecision, MatmulProblem},
    },
    std::MatrixLayout as KernelMatrixLayout,
};

use crate::{
    gemm::{
        problem::{GemmProblem, Precision, problem_for},
        strategy::strategy_for,
    },
    registry::{CorrectnessOutcome, ReferenceMode},
};

/// Stable per-spec seed pair, hashed off `(label, strategy_id, problem_id)`.
/// Returns `(seed_lhs, seed_rhs)` so the two input tensors stay distinct
/// across the same problem.
fn seeds_for(strategy_id: &str, problem_id: &str) -> (u64, u64) {
    use std::hash::{Hash, Hasher};
    fn hash_with(label: &str, strategy_id: &str, problem_id: &str) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        label.hash(&mut h);
        strategy_id.hash(&mut h);
        problem_id.hash(&mut h);
        h.finish()
    }
    (
        hash_with("cubek-correctness-gemm/lhs", strategy_id, problem_id),
        hash_with("cubek-correctness-gemm/rhs", strategy_id, problem_id),
    )
}

pub fn cpu(
    strategy_id: &str,
    problem_id: &str,
    epsilon: f32,
) -> Result<CorrectnessOutcome, String> {
    dispatch(strategy_id, problem_id, epsilon, |client, problem, strategy, sl, sr, eps| {
        validate_strategy(client, problem, strategy, sl, sr, eps, CorrectnessTarget::Cpu)
    })
}

pub fn against_reference(
    strategy_id: &str,
    problem_id: &str,
    mode: ReferenceMode,
    epsilon: f32,
) -> Result<CorrectnessOutcome, String> {
    dispatch(strategy_id, problem_id, epsilon, |client, problem, strategy, sl, sr, eps| {
        let target = match &mode {
            ReferenceMode::WriteReference { path } => CorrectnessTarget::WriteReference(path),
            ReferenceMode::ValidateReference { path } => CorrectnessTarget::ValidateReference(path),
        };
        validate_strategy(client, problem, strategy, sl, sr, eps, target)
    })
}

fn dispatch<F>(
    strategy_id: &str,
    problem_id: &str,
    epsilon: f32,
    run: F,
) -> Result<CorrectnessOutcome, String>
where
    F: FnOnce(
        ComputeClient<TestRuntime>,
        MatmulProblem,
        cubek::matmul::launch::Strategy,
        u64,
        u64,
        f32,
    ) -> CorrectnessReport,
{
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let matmul_problem = build_matmul_problem(&problem);

    let (sl, sr) = seeds_for(strategy_id, problem_id);
    let report = run(client, matmul_problem, strategy, sl, sr, epsilon);
    Ok(into_outcome(report))
}

fn build_matmul_problem(p: &GemmProblem) -> MatmulProblem {
    let global_dtypes = match p.precision {
        Precision::F32 => MatmulElems::from_single_dtype(<f32 as cubecl::frontend::CubePrimitive>::as_type_native_unchecked()).as_global_elems(),
        Precision::F16 => MatmulElems::from_single_dtype(<half::f16 as cubecl::frontend::CubePrimitive>::as_type_native_unchecked()).as_global_elems(),
    };
    MatmulProblem::from_parameters(
        p.m,
        p.n,
        p.k,
        Shape::from(vec![p.b]),
        Shape::from(vec![p.b]),
        kernel_layout(p.lhs_layout),
        kernel_layout(p.rhs_layout),
        KernelMatrixLayout::RowMajor,
        None,
        None,
        global_dtypes,
        AddressType::U32,
    )
}

fn kernel_layout(layout: KernelMatrixLayout) -> KernelMatrixLayout {
    layout
}

fn into_outcome(r: CorrectnessReport) -> CorrectnessOutcome {
    match r {
        CorrectnessReport::Pass => CorrectnessOutcome::Pass,
        CorrectnessReport::Fail(reason) => CorrectnessOutcome::Fail(reason),
        CorrectnessReport::Error(reason) => CorrectnessOutcome::Error(reason),
        CorrectnessReport::WroteReference => CorrectnessOutcome::WroteReference,
    }
}

// Silence unused import warnings without dead-code-flag noise — `shape!` and
// `Precision` keep the imports paired with the surrounding API.
#[allow(dead_code)]
fn _unused() {
    let _ = shape![1];
    let _: Precision = Precision::F32;
}

// Suppress dead-code lints on the marker fn so the crate still builds when
// MatmulPrecision isn't directly referenced.
#[allow(dead_code)]
fn _phantom<MP: MatmulPrecision>() {}
