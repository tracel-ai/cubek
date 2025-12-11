use crate::attention::launcher::test_launch;
use crate::attention::tiling_scheme_ops::*;
use cubecl::{Runtime, TestRuntime};
use cubek_attention::launch::{
    AccumulatorPrecision, AttentionDefinition, AttentionDims, AttentionOptions,
    AttentionPartitionSize, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
};
use cubek_attention::routines::DeviceSettings;

#[test]
fn attention_one_tile_simple() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };

    let definition = AttentionDefinition {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let launch_settings = DeviceSettings::new(&client, &definition);

    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        reuse_key_value: false,
        two_rows_in_array_tile: false,
        line_sizes: launch_settings.line_sizes,
        masked: definition.masked,
        causal: definition.options.causal,
        check_bounds: tiling_scheme.check_bounds(&definition.dims),
    };

    let strategy = strategy(blueprint);

    test_launch(client, definition, strategy)
}

// #[test]
// fn attention_one_partition_several_planes() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE * 2,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_partition_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_problem_smaller_than_one_tile_seq_q_seq_kv_val_dim() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = tiling_scheme.tile_size.seq_q as usize - 1;
//     let seq_kv = tiling_scheme.tile_size.seq_kv as usize - 1;
//     let head_dim = tiling_scheme.tile_size.head_dim as usize;
//     let val_dim = tiling_scheme.tile_size.val_dim as usize - 1;

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_head_dim_oob() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = tiling_scheme.tile_size.head_dim as usize - 1;
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_two_rows_in_array_tile() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         two_rows_in_array_tile: true,
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_one_tile_seqq16() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = 16;
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_one_tile_seqq4() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = 4;
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_seqq2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 2,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_hd2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 2,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_kv2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 2,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_vd2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 2,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_hd2_vd2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 2,
//         val_dim: 2,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_all2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 2,
//         seq_kv: 2,
//         head_dim: 2,
//         val_dim: 2,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_global_iterations_2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_global_iterations_2_kv2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 2,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_kv1_global1_with_oob() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) - 1;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_seqq2_global2_kv2_global2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: 2 * STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_many_planes() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: 15 * STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_kv1_global3_with_oob() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: 2 * STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2 + 1;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_oob_in_q() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 2,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = 1;
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_kv2_with_oob() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 2,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_kv2_causal() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 2,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: true,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_partition_kv2_masked() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 2,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: true,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_stage2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: 2 * STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_stage4() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: 4 * STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_stage2_problem4() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: 2 * STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme) * 2;
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_reuse_key_value() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 2,
//         val_dim: 2,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         reuse_key_value: true,
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_double_row_wise() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         two_rows_in_array_tile: true,
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_one_tile_masked() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: true,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_one_tile_causal() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: true,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_one_tile_masked_causal() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: true,
//         causal: true,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_masked_oob() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) - 1;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: true,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_masked_larger() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: true,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_num_heads_2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 2,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_batch_2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 2,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_batch_2_seqq2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 2,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 2,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_num_heads_2_batch_2() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 2,
//         num_heads: 2,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_num_heads_2_masked() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: 1,
//         val_dim: 1,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let seq_q = elements_in_stage_seq_q(&tiling_scheme);
//     let seq_kv = elements_in_partition_seq_kv(&tiling_scheme);
//     let head_dim = elements_in_partition_head_dim(&tiling_scheme);
//     let val_dim = elements_in_partition_val_dim(&tiling_scheme);

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 2,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: true,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }

// #[test]
// fn attention_huge_problem() {
//     let client = <TestRuntime as Runtime>::client(&Default::default());

//     let seq_q = 128;
//     let seq_kv = 128;
//     let head_dim = 64;
//     let val_dim = 64;

//     let hd = head_dim as u32 / TILE_SIZE.head_dim;

//     let partition_size = AttentionPartitionSize {
//         seq_q: 1,
//         seq_kv: 1,
//         head_dim: hd,
//         val_dim: hd,
//     };
//     let stage_size = AttentionStageSize {
//         seq_q: STAGE_Q_BASE,
//     };
//     let tiling_scheme = AttentionTilingScheme {
//         tile_size: TILE_SIZE,
//         partition_size,
//         stage_size,
//     };

//     let global_dtypes = global_dtypes();
//     let line_sizes = default_line_sizes(&client, global_dtypes.clone(), head_dim, val_dim);

//     let problem = AttentionProblem {
//         batch: 1,
//         num_heads: 1,
//         seq_q,
//         seq_kv,
//         head_dim,
//         val_dim,
//         masked: false,
//         causal: false,
//         global_dtypes: global_dtypes,
//         accumulator_precision: AccumulatorPrecision::default(),
//         line_sizes,
//     };

//     let strategy = strategy(Selection::Inferred(SharedAttentionSettings {
//         tiling_scheme: Some(tiling_scheme),
//         ..Default::default()
//     }));

//     test_launch(client, problem, strategy)
// }
