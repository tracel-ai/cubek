#[test]
fn small_problem_selector() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 8,
            seq_kv: 8,
            head_dim: 8,
            val_dim: 8,
        },
        masked: false,
        global_dtypes: global_dtypes(),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let strategy = inferred_strategy();

    test_launch(client, problem, strategy)
}
