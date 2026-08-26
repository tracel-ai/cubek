mod matmul_unit {
    use cubecl::{TestRuntime, client::ComputeClient};
    use cubek_matmul::{
        definition::MatmulProblem,
        multi_level::{
            Strategy as MultiLevel, definition::BatchMatmulBlueprint, test_only::TestStrategy,
        },
        routine::BlueprintStrategy,
    };

    use crate::harness::{test_matmul_strategy, test_matmul_test_strategy};

    fn launch_simple(c: ComputeClient<TestRuntime>, p: MatmulProblem, bp: BatchMatmulBlueprint) {
        test_matmul_strategy(
            c,
            p,
            MultiLevel::SimpleUnit(BlueprintStrategy::Forced(bp)).into(),
        );
    }

    fn launch_double_buffering(
        c: ComputeClient<TestRuntime>,
        p: MatmulProblem,
        bp: BatchMatmulBlueprint,
    ) {
        test_matmul_strategy(
            c,
            p,
            MultiLevel::DoubleUnit(BlueprintStrategy::Forced(bp)).into(),
        );
    }

    fn launch_interleaved(
        c: ComputeClient<TestRuntime>,
        p: MatmulProblem,
        bp: BatchMatmulBlueprint,
    ) {
        test_matmul_test_strategy(
            c,
            p,
            TestStrategy::Interleaved(BlueprintStrategy::Forced(bp)),
        );
    }

    include!("algorithm.rs");
}
