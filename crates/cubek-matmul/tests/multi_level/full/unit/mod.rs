mod matmul_unit {
    use cubecl::{TestRuntime, client::Client};
    use cubek_matmul::{
        definition::MatmulProblem,
        multi_level::{Strategy as MultiLevel, definition::BatchMatmulBlueprint},
        routine::BlueprintStrategy,
    };

    use crate::harness::test_matmul_strategy;

    fn launch_simple(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
        test_matmul_strategy(
            c,
            p,
            MultiLevel::SimpleUnit(BlueprintStrategy::Forced(bp)).into(),
        );
    }

    fn launch_double_buffering(c: Client, p: MatmulProblem, bp: BatchMatmulBlueprint) {
        test_matmul_strategy(
            c,
            p,
            MultiLevel::DoubleUnit(BlueprintStrategy::Forced(bp)).into(),
        );
    }

    include!("algorithm.rs");
}
