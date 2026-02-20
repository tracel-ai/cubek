#[test]
fn very_small_problem_selector() {
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
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let strategy = inferred_strategy();

    test_launch(client, problem, strategy)
}

#[test]
fn small_problem_selector() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 2048,
            seq_kv: 2048,
            head_dim: 128,
            val_dim: 128,
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let strategy = inferred_strategy();

    test_launch(client, problem, strategy)
}
