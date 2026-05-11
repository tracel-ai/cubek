use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek_test_utils::{HostData, Progress};

use super::problem::{AttentionSpec, build_problem};
use crate::definition::AttentionGlobalTypes;
use crate::eval::cpu_reference::{cpu_reference_result, strategy_result};
use crate::launch::Strategy;

pub struct AttentionCorrectness;

impl cubek_test_utils::Correctness for AttentionCorrectness {
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
