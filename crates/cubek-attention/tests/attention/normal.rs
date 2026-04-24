//! Normal tier: inferred-blueprint tests for each routine family.
//!
//! These tests route through `launch_ref` with `BlueprintStrategy::Inferred`,
//! exercising the selector heuristic for each Strategy variant. The
//! forced-blueprint tiling-scheme sweep lives in the `extended` tier.

use crate::attention::launcher::test_launch;
use cubecl::{
    Runtime, TestRuntime, client::ComputeClient, frontend::CubePrimitive, ir::AddressType,
};
use cubek_attention::{
    definition::{
        AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionOptions,
        AttentionProblem,
    },
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

fn f16_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
    AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(client),
    )
}

fn f32_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
    AttentionGlobalTypes::from_single_float_dtype(
        f32::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(client),
    )
}

fn unit_inferred() -> Strategy {
    Strategy::Unit(BlueprintStrategy::Inferred(()))
}

fn blackbox_accelerated_inferred() -> Strategy {
    Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
        num_planes: 1,
        seq_q: 1,
        seq_kv: 1,
    }))
}

fn problem(
    global_dtypes: AttentionGlobalTypes,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    val_dim: usize,
) -> AttentionProblem {
    AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        },
        masked: false,
        global_dtypes,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    }
}

// TODO remove extended cfg when works on
#[cfg(feature = "extended")]
mod unit {
    use super::*;

    #[test]
    fn f16_very_small() {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        test_launch(
            client.clone(),
            problem(f16_dtypes(&client), 8, 8, 8, 8),
            unit_inferred(),
        )
    }

    #[test]
    fn f16_small() {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        test_launch(
            client.clone(),
            problem(f16_dtypes(&client), 128, 128, 64, 64),
            unit_inferred(),
        )
    }

    #[test]
    fn f16_hd_smaller_than_vd() {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        test_launch(
            client.clone(),
            problem(f16_dtypes(&client), 64, 64, 32, 64),
            unit_inferred(),
        )
    }

    #[test]
    fn f32_very_small() {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        test_launch(
            client.clone(),
            problem(f32_dtypes(&client), 8, 8, 8, 8),
            unit_inferred(),
        )
    }
}

#[test]
fn blackbox_accelerated_f16_very_small() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), 8, 8, 8, 8),
        blackbox_accelerated_inferred(),
    )
}

#[test]
fn blackbox_accelerated_f16_small() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch(
        client.clone(),
        problem(f16_dtypes(&client), 128, 128, 64, 64),
        blackbox_accelerated_inferred(),
    )
}
