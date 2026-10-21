#![cfg(feature = "multi-level")]

use cubecl::{features::Features, prelude::*};
use cubek_matmul::{definition::MatmulElems, multi_level::definition::adjust_dtypes_with_tf32};

#[test]
fn tf32_permission_only_controls_automatic_f32_conversion() {
    let f32_ty = f32::elem_type_native();
    let tf32_ty = tf32::elem_type_native();
    let f16_ty = half::f16::elem_type_native();
    let mut features = Features::default();
    features.types.elem.insert(tf32_ty, Default::default());
    features.types.elem.insert(f16_ty, Default::default());

    for (input, accelerator, allow_tf32, expected) in [
        (f32_ty, true, true, tf32_ty),
        (f32_ty, true, false, f32_ty),
        (f32_ty, false, true, f32_ty),
        (flex32::elem_type_native(), true, false, f16_ty),
        (tf32_ty, true, false, tf32_ty),
    ] {
        let mut elems = MatmulElems::from_single_dtype(input);
        adjust_dtypes_with_tf32(&features, &mut elems, accelerator, allow_tf32);
        let mut expected_elems = MatmulElems::from_single_dtype(input);
        expected_elems.lhs_stage = expected;
        expected_elems.rhs_stage = expected;
        expected_elems.lhs_register = expected;
        expected_elems.rhs_register = expected;
        assert_eq!(elems, expected_elems);
    }

    let mut elems = MatmulElems::from_single_dtype(f32_ty);
    let original = elems.clone();
    adjust_dtypes_with_tf32(&Features::default(), &mut elems, true, true);
    assert_eq!(elems, original, "permission cannot invent device support");
}
