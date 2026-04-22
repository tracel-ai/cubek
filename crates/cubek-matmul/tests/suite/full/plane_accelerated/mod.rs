mod matmul_plane_accelerated {
    mod cmma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::{MatmulProblem, TilingBlueprint},
            launch::{Strategy, test_only::TestStrategy},
            routines::BlueprintStrategy,
        };

        use crate::suite::{test_matmul_strategy, test_matmul_test_strategy};

        fn launch_simple_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SimpleStridedCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SimpleTilewiseCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_barrier_cooperative(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_test_strategy(
                c,
                p,
                TestStrategy::SimpleBarrierCooperativeCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_barrier_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_test_strategy(
                c,
                p,
                TestStrategy::SimpleBarrierCyclicCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_double_buffering_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_double_buffering_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::DoubleTilewiseCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_double_buffering_hybrid(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::DoubleHybridCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_ordered_double_buffering(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_specialized_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_specialized_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SpecializedStridedCmma(BlueprintStrategy::Forced(bp)),
            );
        }

        include!("algorithm.rs");
    }

    mod mma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::{MatmulProblem, TilingBlueprint},
            launch::{Strategy, test_only::TestStrategy},
            routines::BlueprintStrategy,
        };

        use crate::suite::{test_matmul_strategy, test_matmul_test_strategy};

        fn launch_simple_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SimpleCyclicMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SimpleStridedMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SimpleTilewiseMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_barrier_cooperative(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_test_strategy(
                c,
                p,
                TestStrategy::SimpleBarrierCooperativeMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_simple_barrier_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_test_strategy(
                c,
                p,
                TestStrategy::SimpleBarrierCyclicMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_double_buffering_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::DoubleCyclicMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_double_buffering_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::DoubleTilewiseMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_double_buffering_hybrid(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::DoubleHybridMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_ordered_double_buffering(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::OrderedDoubleMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_specialized_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SpecializedCyclicMma(BlueprintStrategy::Forced(bp)),
            );
        }
        fn launch_specialized_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: TilingBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                Strategy::SpecializedStridedMma(BlueprintStrategy::Forced(bp)),
            );
        }

        include!("algorithm.rs");
    }
}
