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
            seq_q: 128,
            seq_kv: 128,
            head_dim: 64,
            val_dim: 64,
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
fn hd_smaller_than_vd_problem_selector() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 64,
            seq_kv: 64,
            head_dim: 32,
            val_dim: 64,
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
fn hd_larger_than_vd_problem_selector() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 64,
            seq_kv: 64,
            head_dim: 64,
            val_dim: 32,
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
