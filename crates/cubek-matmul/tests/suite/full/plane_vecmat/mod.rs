mod matmul_plane_vecmat {
    use cubecl::{TestRuntime, client::ComputeClient};
    use cubek_matmul::{
        definition::{MatmulProblem, TilingBlueprint},
        launch::Strategy,
        routines::BlueprintStrategy,
    };

    use crate::suite::test_matmul_strategy;

    fn launch_simple_cyclic(
        client: ComputeClient<TestRuntime>,
        problem: MatmulProblem,
        bp: TilingBlueprint,
    ) {
        test_matmul_strategy(
            client,
            problem,
            Strategy::SimpleVecMat(BlueprintStrategy::Forced(bp)),
        );
    }

    fn launch_double_buffering_cyclic(
        client: ComputeClient<TestRuntime>,
        problem: MatmulProblem,
        bp: TilingBlueprint,
    ) {
        test_matmul_strategy(
            client,
            problem,
            Strategy::DoubleVecMat(BlueprintStrategy::Forced(bp)),
        );
    }

    include!("algorithm.rs");
}
