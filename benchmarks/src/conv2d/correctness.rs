//! Seeded HostData primitives for the conv2d category.
//!
//! Both `kernel_result` and `reference_result` build the same input bits
//! from `(strategy_id, problem_id, seed_lhs, seed_rhs)` so the two
//! `HostData`s they return are directly comparable.
//!
//! Inputs are laid out NHWC (input) and OHWI (weight). The naive CPU reference
//! is built around this convention; the benchmark `run()` path uses NCHW —
//! correctness here is independent of the timing path.

use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek::{
    convolution::cpu_reference::{ConvSpec, cpu_reference_result, strategy_result},
    matmul::definition::{MatmulElems, MatmulGlobalElems, MatmulPrecision, MatrixPrecision},
};
use cubek_test_utils::HostData;

use crate::conv2d::{
    problem::{Conv2dProblem, problem_for},
    strategy::strategy_for,
};

type LhsG<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
type RhsG<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
type AccG<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Global;

pub fn kernel_result(
    strategy_id: &str,
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let (spec, dtypes) = build_spec_and_dtypes::<half::f16>(&problem);
    strategy_result(client, spec, strategy, dtypes, seed_lhs, seed_rhs)
}

pub fn reference_result(
    problem_id: &str,
    seed_lhs: u64,
    seed_rhs: u64,
) -> Result<HostData, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let (spec, dtypes) = build_spec_and_dtypes::<half::f16>(&problem);
    cpu_reference_result(client, spec, dtypes, seed_lhs, seed_rhs)
}

fn build_spec_and_dtypes<MP: MatmulPrecision>(p: &Conv2dProblem) -> (ConvSpec, MatmulElems) {
    let [n, c_in, h_in, w_in] = p.input_shape;
    let [c_out, _, k_h, k_w] = p.weight_shape;
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: LhsG::<MP>::as_type_native_unchecked().storage_type(),
        rhs: RhsG::<MP>::as_type_native_unchecked().storage_type(),
        out: AccG::<MP>::as_type_native_unchecked().storage_type(),
    });
    let spec = ConvSpec {
        batches: n,
        in_h: h_in,
        in_w: w_in,
        channels: c_in,
        out_channels: c_out,
        args: p.args.clone(),
        kernel_size: [k_h, k_w],
    };
    (spec, dtypes)
}
