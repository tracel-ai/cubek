mod matmul_plane_vecmat {
    use cubecl::{TestRuntime, client::Client};
    use cubek_matmul::{
        definition::MatmulProblem,
        multi_level::{Strategy as MultiLevel, definition::BatchMatmulBlueprint},
        routine::BlueprintStrategy,
        strategy::Strategy,
    };

    use crate::harness::test_matmul_strategy;

    fn launch_simple_cyclic(client: Client, problem: MatmulProblem, bp: BatchMatmulBlueprint) {
        test_matmul_strategy(
            client,
            problem,
            MultiLevel::SimpleVecMat(BlueprintStrategy::Forced(bp)).into(),
        );
    }

    fn launch_double_buffering_cyclic(
        client: Client,
        problem: MatmulProblem,
        bp: BatchMatmulBlueprint,
    ) {
        test_matmul_strategy(
            client,
            problem,
            MultiLevel::DoubleVecMat(BlueprintStrategy::Forced(bp)).into(),
        );
    }

    include!("algorithm.rs");
}
