mod matmul_plane_accelerated {
    mod cmma {
        use cubecl::{TestRuntime, client::Client};
        use cubek_matmul::{
            definition::MatmulProblem,
            multi_level::{Strategy as MultiLevel, definition::BatchMatmulBlueprint},
            routine::BlueprintStrategy,
        };

        use crate::harness::test_matmul_strategy;

        fn launch_simple_cyclic(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_cyclic(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_ordered_double_buffering(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_cyclic(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        include!("algorithm.rs");
    }

    mod mma {
        use cubecl::{TestRuntime, client::Client};
        use cubek_matmul::{
            definition::MatmulProblem,
            multi_level::{Strategy as MultiLevel, definition::BatchMatmulBlueprint},
            routine::BlueprintStrategy,
        };

        use crate::harness::test_matmul_strategy;

        fn launch_simple_cyclic(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleCyclicMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_cyclic(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleCyclicMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_ordered_double_buffering(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::OrderedDoubleMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_cyclic(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedCyclicMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        include!("algorithm.rs");
    }
}
