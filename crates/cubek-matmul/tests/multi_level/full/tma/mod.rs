mod matmul_tma {
    mod cmma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::MatmulProblem,
            multi_level::{Strategy as MultiLevel, definition::BatchMatmulBlueprint},
            routine::BlueprintStrategy,
            strategy::Strategy,
        };

        use crate::harness::test_matmul_strategy;

        fn launch_simple_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleTmaCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleTmaCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedTmaCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }

        include!("algorithm.rs");
    }

    mod mma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::MatmulProblem, multi_level::definition::BatchMatmulBlueprint,
            routine::BlueprintStrategy, strategy::Strategy,
        };

        use crate::harness::test_matmul_strategy;

        fn launch_simple_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleTmaMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleTmaMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedTmaMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }

        include!("algorithm.rs");
    }
}
