mod matmul_tma {
    mod cmma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::{MatmulProblem, TilingBlueprint},
            launch::Strategy,
            routines::BlueprintStrategy,
        };

        use crate::suite::test_matmul_strategy;

        fn launch_simple_tma(c: ComputeClient<TestRuntime>, p: MatmulProblem, bp: TilingBlueprint) {
            test_matmul_strategy(c, p, Strategy::SimpleTmaCmma(BlueprintStrategy::Forced(bp)));
        }
        fn launch_double_buffering_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(c, p, Strategy::DoubleTmaCmma(BlueprintStrategy::Forced(bp)));
        }
        fn launch_specialized_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SpecializedTmaCmma(BlueprintStrategy::Forced(bp)),
            );
        }

        include!("algorithm.rs");
    }

    mod mma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::{MatmulProblem, TilingBlueprint},
            launch::Strategy,
            routines::BlueprintStrategy,
        };

        use crate::suite::test_matmul_strategy;

        fn launch_simple_tma(c: ComputeClient<TestRuntime>, p: MatmulProblem, bp: TilingBlueprint) {
            test_matmul_strategy(c, p, Strategy::SimpleTmaMma(BlueprintStrategy::Forced(bp)));
        }
        fn launch_double_buffering_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(c, p, Strategy::DoubleTmaMma(BlueprintStrategy::Forced(bp)));
        }
        fn launch_specialized_tma(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SpecializedTmaMma(BlueprintStrategy::Forced(bp)),
            );
        }

        include!("algorithm.rs");
    }
}
