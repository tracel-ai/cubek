//! Seeded HostData primitives for the attention category.
//!
//! Both methods build the same input bits from `(strategy, spec, seeds[0..2])`
//! so the two `HostData`s they return are directly comparable.

use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek::attention::{
    cpu_reference::{cpu_reference_result, strategy_result},
    definition::AttentionGlobalTypes,
    launch::Strategy,
};
use cubek_test_utils::{HostData, Progress};

use crate::attention::problem::{AttentionSpec, build_problem};

pub struct AttentionCorrectness;

impl crate::registry::Correctness for AttentionCorrectness {
    type Problem = AttentionSpec;
    type Strategy = Strategy;

    fn kernel_result(
        &self,
        strategy: &Strategy,
        spec: &AttentionSpec,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dtypes = AttentionGlobalTypes::from_single_float_dtype(
            half::f16::as_type_native_unchecked(),
            AttentionGlobalTypes::mask_dtype(&client),
        );
        strategy_result(
            client,
            build_problem(spec, dtypes),
            strategy.clone(),
            seeds[0],
            seeds[1],
        )
    }

    fn reference_result(
        &self,
        spec: &AttentionSpec,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = <TestRuntime as Runtime>::Device::default();
        let client = <TestRuntime as Runtime>::client(&device);
        let dtypes = AttentionGlobalTypes::from_single_float_dtype(
            half::f16::as_type_native_unchecked(),
            AttentionGlobalTypes::mask_dtype(&client),
        );
        cpu_reference_result(
            client,
            build_problem(spec, dtypes),
            seeds[0],
            seeds[1],
            progress,
        )
    }
}
