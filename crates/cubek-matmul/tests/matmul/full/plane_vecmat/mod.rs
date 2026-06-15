mod matmul_plane_vecmat {
    use cubecl::{TestRuntime, client::ComputeClient};
    use cubek_matmul::{
        definition::{BatchMatmulBlueprint, MatmulProblem},
        routines::BlueprintStrategy,
        strategy::Strategy,
    };

    use crate::matmul::test_matmul_strategy;

    fn launch_simple_cyclic(
        client: ComputeClient<TestRuntime>,
        problem: MatmulProblem,
        bp: BatchMatmulBlueprint,
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
        bp: BatchMatmulBlueprint,
    ) {
        test_matmul_strategy(
            client,
            problem,
            Strategy::DoubleVecMat(BlueprintStrategy::Forced(bp)),
        );
    }

    include!("algorithm.rs");
}
