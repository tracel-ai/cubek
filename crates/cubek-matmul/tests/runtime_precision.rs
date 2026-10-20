#![cfg(all(feature = "multi-level", not(target_family = "wasm")))]

use std::hash::{DefaultHasher, Hash, Hasher};

use cubecl::{
    features::Features,
    prelude::*,
    zspace::{shape, strides},
};
use cubek_matmul::{
    definition::MatmulElems, multi_level::definition::adjust_dtypes_with_tf32,
    tune_key::MatmulAutotuneKey,
};
use cubek_std::config::{CubekConfig, MatmulConfig, RuntimeConfig};

#[test]
fn policy_separates_persisted_autotune_keys() {
    let run = |allow_tf32: bool| {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "policy_child", "--nocapture"])
            .env("CUBEK_TEST_ALLOW_TF32", allow_tf32.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let serialized = stdout
            .lines()
            .find_map(|line| line.strip_prefix("KEY="))
            .expect("child must report its serialized autotune key");
        serde_json::from_str::<MatmulAutotuneKey>(serialized).unwrap()
    };
    let enabled = run(true);
    let disabled = run(false);
    assert!(enabled.definition.allow_tf32);
    assert!(!disabled.definition.allow_tf32);
    assert_ne!(enabled, disabled);
    let hash = |key: &MatmulAutotuneKey| {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    };
    assert_ne!(hash(&enabled), hash(&disabled));
}

#[test]
fn policy_child() {
    let Ok(setting) = std::env::var("CUBEK_TEST_ALLOW_TF32") else {
        return;
    };
    let allow_tf32 = setting.parse().unwrap();
    CubekConfig::set(CubekConfig {
        matmul: MatmulConfig { allow_tf32 },
    });
    let f32_ty = f32::elem_type_native();
    let tf32_ty = tf32::elem_type_native();
    let f16_ty = half::f16::elem_type_native();
    let mut features = Features::default();
    features.types.elem.insert(tf32_ty, Default::default());
    features.types.elem.insert(f16_ty, Default::default());
    for (input, accelerator, expected) in [
        (f32_ty, true, if allow_tf32 { tf32_ty } else { f32_ty }),
        (f32_ty, false, f32_ty),
        (flex32::elem_type_native(), true, f16_ty),
        (
            flex32::elem_type_native(),
            false,
            flex32::elem_type_native(),
        ),
        (tf32_ty, true, tf32_ty),
        (f16_ty, true, f16_ty),
        (
            half::bf16::elem_type_native(),
            true,
            half::bf16::elem_type_native(),
        ),
    ] {
        let mut elems = MatmulElems::from_single_dtype(input);
        adjust_dtypes_with_tf32(
            &features,
            &mut elems,
            accelerator,
            MatmulConfig::get().allow_tf32,
        );
        assert_eq!(elems.lhs_stage, expected);
        assert_eq!(elems.rhs_stage, expected);
        assert_eq!(elems.lhs_register, expected);
        assert_eq!(elems.rhs_register, expected);
        assert_eq!(elems.lhs_global, input);
        assert_eq!(elems.rhs_global, input);
        assert_eq!(elems.acc_global, input);
        assert_eq!(elems.acc_stage, input);
        assert_eq!(elems.acc_register, input);
    }
    let mut elems = MatmulElems::from_single_dtype(f32_ty);
    // Convolution can keep its existing precision selection independently of
    // the matmul setting, using the same explicit helper.
    adjust_dtypes_with_tf32(&features, &mut elems, true, true);
    assert_eq!(elems.lhs_register, tf32_ty);
    let mut elems = MatmulElems::from_single_dtype(f32_ty);
    let original = elems.clone();
    adjust_dtypes_with_tf32(&Features::default(), &mut elems, true, allow_tf32);
    assert_eq!(elems, original, "permission cannot invent device support");
    elems.rhs_global = f16_ty;
    let mixed = elems.clone();
    adjust_dtypes_with_tf32(&features, &mut elems, true, allow_tf32);
    assert_eq!(elems, mixed, "mixed input types must not be narrowed");

    let key = MatmulAutotuneKey::from_parts(
        &shape![32, 64],
        &shape![64, 32],
        &strides![64, 1],
        &strides![32, 1],
        cubecl::zspace::Tiling::UNTILED,
        cubecl::zspace::Tiling::UNTILED,
        f32_ty,
        f32_ty,
        f32_ty,
        None,
        None,
    );
    println!("KEY={}", serde_json::to_string(&key).unwrap());
}
