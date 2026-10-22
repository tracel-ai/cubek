use cubecl::{
    prelude::*,
    zspace::{Tiling, shape, strides},
};
use cubek_matmul::{
    definition::MatmulElems, launch::launch_ref, strategy::Strategy, tune_key::MatmulAutotuneKey,
};
use cubek_std::{
    InputBinding,
    config::{CubekConfig, MatmulConfig, RuntimeConfig},
};

// A separate integration-test binary keeps the process-wide setting independent
// of the main suite's defaults. Run with any supported CubeCL test backend.
#[test]
fn auto_preserves_f32_precision_when_tf32_is_disabled() {
    CubekConfig::set(CubekConfig {
        matmul: MatmulConfig { allow_tf32: false },
    });
    let client = cubecl::test_device().client();
    let ty = f32::elem_type_native();
    // Exact in F32 but rounds to 1.0 in TF32, exposing accidental narrowing.
    let value = 1.0f32 + 1.0 / 4096.0;
    let binding = |value| TensorBinding {
        handle: client
            .create_from_slice(bytemuck::cast_slice(&[value; 32 * 32]))
            .binding(),
        shape: shape![32, 32],
        strides: strides![32, 1],
        tiling: Tiling::UNTILED,
    };
    let lhs = binding(value);
    let rhs = binding(1.0f32);
    let output = client.empty(32 * 32 * 4);
    let out = TensorBinding {
        handle: output.clone().binding(),
        shape: shape![32, 32],
        strides: strides![32, 1],
        tiling: Tiling::UNTILED,
    };
    let key = MatmulAutotuneKey::from_parts(
        &lhs.shape,
        &rhs.shape,
        &lhs.strides,
        &rhs.strides,
        lhs.tiling,
        rhs.tiling,
        ty,
        ty,
        ty,
        None,
        None,
    );
    assert!(!key.definition.allow_tf32);

    let mut elems = MatmulElems::from_single_dtype(ty);
    launch_ref(
        &Strategy::Auto,
        &client,
        InputBinding::Normal(lhs, ty),
        InputBinding::Normal(rhs, ty),
        out,
        &mut elems,
    )
    .unwrap();
    let bytes = client.read_one(output).unwrap();
    let actual: &[f32] = bytemuck::cast_slice(&bytes);
    assert!(actual.iter().all(|&v| v == 32.0 * value));
    assert_eq!(elems.lhs_register, ty);
    assert_eq!(elems.rhs_register, ty);
}
