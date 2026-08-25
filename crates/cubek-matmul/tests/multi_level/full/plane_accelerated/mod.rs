mod matmul_plane_accelerated {
    mod cmma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::MatmulProblem,
            multi_level::{
                Strategy as MultiLevel, definition::BatchMatmulBlueprint, test_only::TestStrategy,
            },
            routine::BlueprintStrategy,
        };

        use crate::harness::{test_matmul_strategy, test_matmul_test_strategy};

        fn launch_simple_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_simple_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleStridedCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_simple_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleTilewiseCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_simple_barrier_cooperative(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
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
            bp: BatchMatmulBlueprint,
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
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleTilewiseCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_hybrid(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleHybridCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_ordered_double_buffering(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedStridedCmma(BlueprintStrategy::Forced(bp)).into(),
            );
        }

        include!("algorithm.rs");
    }

    mod mma {
        use cubecl::{TestRuntime, client::ComputeClient};
        use cubek_matmul::{
            definition::MatmulProblem,
            multi_level::{
                Strategy as MultiLevel, definition::BatchMatmulBlueprint, test_only::TestStrategy,
            },
            routine::BlueprintStrategy,
        };

        use crate::harness::{test_matmul_strategy, test_matmul_test_strategy};

        fn launch_simple_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleCyclicMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_simple_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleStridedMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_simple_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SimpleTilewiseMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_simple_barrier_cooperative(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
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
            bp: BatchMatmulBlueprint,
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
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleCyclicMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_tilewise(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleTilewiseMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_double_buffering_hybrid(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::DoubleHybridMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_ordered_double_buffering(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::OrderedDoubleMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_cyclic(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedCyclicMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }
        fn launch_specialized_strided(
            c: ComputeClient<TestRuntime>,
            p: MatmulProblem,
            bp: BatchMatmulBlueprint,
        ) {
            test_matmul_strategy(
                c,
                p,
                MultiLevel::SpecializedStridedMma(BlueprintStrategy::Forced(bp)).into(),
            );
        }

        include!("algorithm.rs");
    }
}
