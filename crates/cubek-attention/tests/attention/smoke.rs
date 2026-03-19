use crate::attention::launcher::test_launch;
use cubecl::{Runtime, TestRuntime, ir::AddressType, prelude::CubePrimitive as _};
use cubek_attention::{
    definition::{
        AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionOptions,
        AttentionProblem,
    },
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

#[test]
fn small_blackbox_accelerated() {
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
        global_dtypes: AttentionGlobalTypes::from_single_float_dtype(
            half::f16::as_type_native_unchecked(),
            AttentionGlobalTypes::mask_dtype(&client),
        ),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
        address_type: AddressType::default(),
    };

    let strategy =
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
            num_planes: 1,
            seq_q: 1,
            seq_kv: 1,
        }));

    test_launch(client, problem, strategy)
}
