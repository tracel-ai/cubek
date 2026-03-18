use crate::attention::launcher::test_launch;
use crate::attention::tiling_scheme_ops::*;
use cubecl::ir::AddressType;
use cubecl::{Runtime, TestRuntime};
use cubek_attention::definition::{
    AccumulatorPrecision, AttentionDims, AttentionOptions, AttentionPartitionSize,
    AttentionProblem, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
};
use cubek_attention::routines::DeviceSettings;

#[test]
fn one_tile_simple() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let launch_settings = DeviceSettings::new(&client, &problem);

    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };

    let strategy = forced_strategy(blueprint);

    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_hd_neq_vd() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let tiling_scheme = AttentionTilingScheme {
        tile_size: AttentionTileSize {
            seq_q: 8,
            seq_kv: 8,
            head_dim: 4,
            val_dim: 8,
        },
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

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let launch_settings = DeviceSettings::new(&client, &problem);

    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };

    let strategy = forced_strategy(blueprint);

    test_launch(client, problem, strategy)
}

#[test]
fn partition_hd_neq_vd() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 2,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let launch_settings = DeviceSettings::new(&client, &problem);

    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };

    let strategy = forced_strategy(blueprint);

    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_several_planes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 2,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage() * 2,
        },
    };

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let launch_settings = DeviceSettings::new(&client, &problem);

    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };

    let strategy = forced_strategy(blueprint);

    test_launch(client, problem, strategy)
}

#[test]
fn problem_smaller_than_one_tile_seq_q_seq_kv_val_dim() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_q = tiling_scheme.tile_size.seq_q as usize - 1;
    let seq_kv = tiling_scheme.tile_size.seq_kv as usize - 1;
    let head_dim = tiling_scheme.tile_size.head_dim as usize;
    let val_dim = tiling_scheme.tile_size.val_dim as usize - 1;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn two_rows_in_array_tile() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: true,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_seqq16() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_q = 16;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_seqq4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_q = 4;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn seqq2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 2,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn kv2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 2,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn hd2_vd2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 2,
            val_dim: 2,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn all2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 2,
            seq_kv: 2,
            head_dim: 2,
            val_dim: 2,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn global_iterations_2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn global_iterations_2_kv2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 2,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_kv1_global1_with_oob() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) - 1;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_seqq2_global2_kv2_global2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: 2 * minimal_seq_q_stage(),
        },
    };
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_many_planes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: 10 * minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_kv1_global3_with_oob() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: 2 * minimal_seq_q_stage(),
        },
    };
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2 + 1;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_oob_in_q_with_batches() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 2,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 2,
            num_heads: 2,
            seq_q: 1,
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_kv2_with_oob() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 2,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme) - 1,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_kv2_causal() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 2,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: true,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn partition_kv2_masked() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 2,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: true,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn stage2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: 2 * minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn stage4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: 4 * minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn stage2_problem4() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: 2 * minimal_seq_q_stage(),
        },
    };
    let seq_q = elements_in_stage_seq_q(&tiling_scheme) * 2;
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn double_row_wise() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: true,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_masked() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: true,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_causal() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: true,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn one_tile_masked_causal() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: true,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: true,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn masked_oob() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) - 1;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: true,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn masked_larger() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let seq_kv = elements_in_partition_seq_kv(&tiling_scheme) * 2;
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv,
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: true,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn num_heads_2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 2,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn batch_2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 2,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn batch_2_seqq2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 2,
            seq_kv: 1,
            head_dim: 1,
            val_dim: 1,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 2,
            num_heads: 1,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn num_heads_2_batch_2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 2,
            num_heads: 2,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn num_heads_2_masked() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
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
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 2,
            seq_q: elements_in_stage_seq_q(&tiling_scheme),
            seq_kv: elements_in_partition_seq_kv(&tiling_scheme),
            head_dim: elements_in_partition_head_dim(&tiling_scheme),
            val_dim: elements_in_partition_val_dim(&tiling_scheme),
        },
        masked: true,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn huge_problem() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let seq_q = 128;
    let seq_kv = 128;
    let head_dim = 64;
    let val_dim = 64;
    let hd = head_dim as u32 / tile_size(&client, global_dtypes(&client)).head_dim;
    let tiling_scheme = AttentionTilingScheme {
        tile_size: tile_size(&client, global_dtypes(&client)),
        partition_size: AttentionPartitionSize {
            seq_q: 1,
            seq_kv: 1,
            head_dim: hd,
            val_dim: hd,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}

#[test]
fn causal_several_heads() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let head_val_dim = 32;

    let tile_size = tile_size(&client, global_dtypes(&client));
    let tiling_scheme = AttentionTilingScheme {
        tile_size,
        partition_size: AttentionPartitionSize {
            seq_q: 2,
            seq_kv: 2,
            head_dim: head_val_dim / tile_size.head_dim,
            val_dim: head_val_dim / tile_size.val_dim,
        },
        stage_size: AttentionStageSize {
            seq_q: minimal_seq_q_stage(),
        },
    };
    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 2,
            seq_q: 16,
            seq_kv: 16,
            head_dim: head_val_dim as usize,
            val_dim: head_val_dim as usize,
        },
        masked: false,
        global_dtypes: global_dtypes(&client),
        options: AttentionOptions {
            causal: true,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };
    let launch_settings = DeviceSettings::new(&client, &problem);
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint {},
        tiling_scheme,
        plane_dim: launch_settings.plane_dim,
        two_rows_in_array_tile: false,
        vector_sizes: launch_settings.vector_sizes,
        masked: problem.masked,
        causal: problem.options.causal,
        check_bounds: tiling_scheme.check_bounds(&problem.dims),
    };
    let strategy = forced_strategy(blueprint);
    test_launch(client, problem, strategy)
}
