use cubecl::prelude::*;
use cubecl::zspace::shape;
use cubek_test_utils::{DataKind, StridedLayout, TestInput, assert_equals_approx};

#[test]
fn builder_matches_constructor() {
    let client = cubecl::test_device().client();

    // Builder form: defaults to f32 + RowMajor; finalizer picks the data kind.
    let from_builder = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();

    // Equivalent construction with the long-form `TestInput::new`.
    let from_new = TestInput::new(
        client.clone(),
        shape![2, 3],
        f32::elem_type_native(),
        StridedLayout::RowMajor,
        DataKind::Arange { scale: None },
    )
    .f32_host_data();

    assert_equals_approx(&from_builder, &from_new, 0.0)
        .as_test_outcome()
        .enforce();
}

#[test]
fn builder_overrides_stride_and_dtype() {
    let client = cubecl::test_device().client();

    // Builder with explicit stride override should match the constructor.
    let from_builder = TestInput::builder(client.clone(), shape![2, 3])
        .layout(StridedLayout::ColMajor)
        .arange()
        .f32_host_data();

    let from_new = TestInput::new(
        client.clone(),
        shape![2, 3],
        f32::elem_type_native(),
        StridedLayout::ColMajor,
        DataKind::Arange { scale: None },
    )
    .f32_host_data();

    assert_equals_approx(&from_builder, &from_new, 0.0)
        .as_test_outcome()
        .enforce();
}

#[test]
fn builder_linspace_produces_evenly_spaced_values() {
    let client = cubecl::test_device().client();

    // 5 evenly-spaced values from 0.0 to 1.0 → step = 0.25.
    // Shape [1, 5] because RowMajor requires ≥ 2 dimensions.
    let actual = TestInput::builder(client.clone(), shape![1, 5])
        .linspace(0.0, 1.0)
        .f32_host_data();

    let expected = TestInput::builder(client.clone(), shape![1, 5])
        .custom(vec![0.0, 0.25, 0.5, 0.75, 1.0])
        .f32_host_data();

    assert_equals_approx(&actual, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}

#[test]
fn builder_normal_distribution_within_statistical_bounds() {
    // `cubek_random::seed` is a global, so we don't pin bit-exact reproduction
    // (other tests racing on the same global would flake). Instead check that
    // the distribution's *empirical* mean / std stay within sample-size bounds.
    let client = cubecl::test_device().client();

    // Shape [1, N] because RowMajor requires ≥ 2 dimensions.
    let n: usize = 4096;
    let host = TestInput::builder(client.clone(), shape![1, n])
        .normal(11, /* mean */ 0.0, /* std */ 1.0)
        .f32_host_data();

    let values: Vec<f32> = (0..n).map(|i| host.get_f32(&[0, i])).collect();
    let mean = values.iter().copied().sum::<f32>() / n as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();

    // SE of the mean is std/sqrt(n) ≈ 1/64 ≈ 0.016. Use 0.1 as a safe bound.
    assert!(mean.abs() < 0.1, "mean drifted: {mean}");
    assert!((std - 1.0).abs() < 0.1, "std drifted: {std}");
}

#[test]
fn builder_uniform_values_in_range() {
    // Range/property check rather than seed-stability: see the note on the
    // normal test above.
    let client = cubecl::test_device().client();

    let host = TestInput::builder(client.clone(), shape![4, 4])
        .uniform(7, -1.0, 1.0)
        .f32_host_data();

    for i in 0..4 {
        for j in 0..4 {
            let v = host.get_f32(&[i, j]);
            assert!(
                (-1.0..=1.0).contains(&v),
                "uniform value out of range at [{i},{j}]: {v}"
            );
        }
    }
}
