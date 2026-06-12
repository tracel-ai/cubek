mod matmul_unit {
    use cubecl::{TestRuntime, client::ComputeClient};
    use cubek_matmul::{
        definition::{BatchMatmulBlueprint, MatmulProblem},
        routines::BlueprintStrategy,
        strategy::{Strategy, test_only::TestStrategy},
    };

    use crate::matmul::{test_matmul_strategy, test_matmul_test_strategy};

    fn launch_simple(c: ComputeClient<TestRuntime>, p: MatmulProblem, bp: BatchMatmulBlueprint) {
        test_matmul_strategy(c, p, Strategy::SimpleUnit(BlueprintStrategy::Forced(bp)));
    }

    fn launch_double_buffering(
        c: ComputeClient<TestRuntime>,
        p: MatmulProblem,
        bp: BatchMatmulBlueprint,
    ) {
        test_matmul_strategy(c, p, Strategy::DoubleUnit(BlueprintStrategy::Forced(bp)));
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
