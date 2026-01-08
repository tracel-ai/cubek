// Tests for Inferred blueprint strategy with half-precision types
//
// Regression tests for the tile size alignment bug where hardcoded tile_size=4
// caused assertion failures with f16/bf16 (which have line_size=8 on CUDA).

use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime};
use cubek_attention::definition::{
    AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionLineSizes,
    AttentionOptions, AttentionProblem,
};
use cubek_attention::launch::BlueprintStrategy;
use cubek_attention::routines::{unit::UnitRoutine, DeviceSettings, Routine};

/// Test Inferred strategy with f16 produces aligned tile sizes
///
/// Before the fix, this would produce tile_size=4 which fails the kernel assertion
/// `num_cols % line_size == 0` when line_size=8 for f16.
#[test]
fn unit_inferred_f16_tile_alignment() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let global_dtypes =
        AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 64,
            seq_kv: 64,
            head_dim: 64,
            val_dim: 64,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let device_settings = DeviceSettings::new(&client, &problem);
    let strategy = BlueprintStrategy::Inferred(());

    let result = UnitRoutine::prepare(&problem, &device_settings, strategy);
    assert!(result.is_ok(), "f16 Inferred strategy failed: {:?}", result.err());

    // Verify tile sizes are aligned with line sizes
    let launch_info = result.unwrap();
    let tile = &launch_info.blueprint.tiling_scheme.tile_size;
    let lines = &launch_info.blueprint.line_sizes;

    assert!(
        tile.head_dim % lines.query.max(lines.key) as u32 == 0,
        "tile.head_dim ({}) not aligned with line_size ({})",
        tile.head_dim,
        lines.query.max(lines.key)
    );
    assert!(
        tile.val_dim % lines.value.max(lines.out) as u32 == 0,
        "tile.val_dim ({}) not aligned with line_size ({})",
        tile.val_dim,
        lines.value.max(lines.out)
    );
}

/// Test Inferred strategy with bf16 produces aligned tile sizes
#[test]
fn unit_inferred_bf16_tile_alignment() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let global_dtypes =
        AttentionGlobalTypes::from_single_dtype(half::bf16::as_type_native_unchecked());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 64,
            seq_kv: 64,
            head_dim: 64,
            val_dim: 64,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let device_settings = DeviceSettings::new(&client, &problem);
    let strategy = BlueprintStrategy::Inferred(());

    let result = UnitRoutine::prepare(&problem, &device_settings, strategy);
    assert!(result.is_ok(), "bf16 Inferred strategy failed: {:?}", result.err());

    let launch_info = result.unwrap();
    let tile = &launch_info.blueprint.tiling_scheme.tile_size;
    let lines = &launch_info.blueprint.line_sizes;

    assert!(
        tile.head_dim % lines.query.max(lines.key) as u32 == 0,
        "tile.head_dim ({}) not aligned with line_size ({})",
        tile.head_dim,
        lines.query.max(lines.key)
    );
}

/// Test Inferred strategy with f32 still works (regression test)
#[test]
fn unit_inferred_f32_tile_alignment() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let global_dtypes = AttentionGlobalTypes::from_single_dtype(f32::as_type_native_unchecked());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 64,
            seq_kv: 64,
            head_dim: 64,
            val_dim: 64,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    let device_settings = DeviceSettings::new(&client, &problem);
    let strategy = BlueprintStrategy::Inferred(());

    let result = UnitRoutine::prepare(&problem, &device_settings, strategy);
    assert!(result.is_ok(), "f32 Inferred strategy failed: {:?}", result.err());
}

/// Test that Inferred strategy produces aligned tiles when line_size=8 (CUDA f16/bf16)
///
/// This test simulates CUDA's line_size=8 for half-precision types.
/// Before the fix, tile_size was hardcoded to 4, failing `num_cols % line_size == 0`.
#[test]
fn unit_inferred_cuda_line_size_8_tile_alignment() {
    let global_dtypes =
        AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked());

    let problem = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q: 64,
            seq_kv: 64,
            head_dim: 64,
            val_dim: 64,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    // Simulate CUDA line_size=8 for half-precision
    let device_settings = DeviceSettings {
        plane_dim: 32,
        line_sizes: AttentionLineSizes {
            query: 8,
            key: 8,
            value: 8,
            mask: 8,
            out: 8,
        },
    };
    let strategy = BlueprintStrategy::Inferred(());

    let result = UnitRoutine::prepare(&problem, &device_settings, strategy);
    assert!(
        result.is_ok(),
        "f16 with line_size=8 Inferred strategy failed: {:?}",
        result.err()
    );

    // Verify tile sizes are aligned with line sizes
    let launch_info = result.unwrap();
    let tile = &launch_info.blueprint.tiling_scheme.tile_size;

    assert!(
        tile.head_dim >= 8 && tile.head_dim % 8 == 0,
        "tile.head_dim ({}) not aligned with line_size 8",
        tile.head_dim
    );
    assert!(
        tile.val_dim >= 8 && tile.val_dim % 8 == 0,
        "tile.val_dim ({}) not aligned with line_size 8",
        tile.val_dim
    );
}
