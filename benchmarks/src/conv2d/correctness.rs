//! Seeded HostData primitives for the conv2d category.
//!
//! Both methods build the same input bits from `(strategy, problem, seeds[0..2])`
//! so the two `HostData`s they return are directly comparable.
//!
//! Inputs are laid out NHWC (input) and OHWI (weight). The naive CPU reference
//! is built around this convention; the benchmark `bench()` path uses NCHW —
//! correctness here is independent of the timing path.

use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek::{
    convolution::{
        Strategy,
        cpu_reference::{ConvSpec, cpu_reference_result, strategy_result},
    },
    matmul::definition::{MatmulElems, MatmulGlobalElems, MatmulPrecision, MatrixPrecision},
};
use cubek_test_utils::{HostData, Progress};

use crate::conv2d::problem::Conv2dProblem;

type LhsG<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
type RhsG<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
type AccG<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Global;

pub struct Conv2dCorrectness;

impl crate::registry::Correctness for Conv2dCorrectness {
    type Problem = Conv2dProblem;
    type Strategy = Strategy;

    fn kernel_result(
        &self,
        strategy: &Strategy,
        problem: &Conv2dProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let (spec, dtypes) = build_spec_and_dtypes::<half::f16>(problem);
        strategy_result(client, spec, strategy.clone(), dtypes, seeds[0], seeds[1])
    }

    fn reference_result(
        &self,
        problem: &Conv2dProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let (spec, dtypes) = build_spec_and_dtypes::<half::f16>(problem);
        cpu_reference_result(client, spec, dtypes, seeds[0], seeds[1], progress)
    }
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
