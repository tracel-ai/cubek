use cubecl::{
    frontend::CubePrimitive,
    zspace::shape,
    {Runtime, TestRuntime},
};
use cubek_test_utils::{
    DataKind, HostData, HostDataType, StrideSpec, TestInput, ValidationResult,
    assert_equals_approx, assert_equals_approx_in_slice, print_tensor,
};

#[test]
fn eye_handle_row_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = [2, 3];

    let handle = TestInput::builder(client.clone(), shape).eye().generate();

    let expected = TestInput::builder(client.clone(), [2, 3])
        .custom(vec![1., 0., 0., 0., 1., 0.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn eye_handle_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = [2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .stride(StrideSpec::ColMajor)
        .eye()
        .generate();

    let expected = TestInput::builder(client.clone(), [2, 3])
        .custom(vec![1., 0., 0., 0., 1., 0.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_row_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = shape![2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .arange()
        .generate();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = shape![2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .stride(StrideSpec::ColMajor)
        .arange()
        .generate();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn custom_handle_row_major_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let contiguous_data = [9., 8., 7., 6., 5., 4.].to_vec();

    let (_, row_major) = TestInput::builder(client.clone(), shape![2, 3])
        .custom(contiguous_data.clone())
        .generate_with_f32_host_data();

    let (_, col_major) = TestInput::builder(client.clone(), shape![2, 3])
        .stride(StrideSpec::ColMajor)
        .custom(contiguous_data)
        .generate_with_f32_host_data();

    assert_equals_approx(&col_major, &row_major, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_row_major_slice() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = shape![2, 3];

    // Create an "actual" tensor where the second row differs
    let actual_data = vec![0., 1., 2., 9., 9., 9.]; // last 3 elements differ
    let actual = TestInput::builder(client.clone(), shape.clone())
        .custom(actual_data)
        .f32_host_data();

    // Expected tensor
    let expected_data = vec![0., 1., 2., 3., 4., 5.];
    let expected = TestInput::builder(client.clone(), shape)
        .custom(expected_data)
        .f32_host_data();

    assert_equals_approx_in_slice(&actual, &expected, 0.001, vec![0..1, 0..3])
        .as_test_outcome()
        .enforce();
}

#[test]
fn fail_message_contains_aggregate_stats_and_examples() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // 6 elements, 3 of which are wrong by 1.0 each.
    let actual = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 4., 5., 6.])
        .f32_host_data();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    let result = assert_equals_approx(&actual, &expected, 0.001);
    let reason = match result {
        ValidationResult::Fail(r) => r,
        other => panic!("expected Fail, got {other:?}"),
    };

    assert!(
        reason.contains("3/6 elements mismatched"),
        "missing mismatch count, got: {reason}"
    );
    assert!(
        reason.contains("max |Δ|="),
        "missing max delta, got: {reason}"
    );
    // Worst index has the largest |delta|. All deltas are 1.0, so the *first*
    // wrong index is recorded as the worst (record_delta only updates on '>').
    assert!(
        reason.contains("worst at [1, 0]"),
        "missing worst index, got: {reason}"
    );

    // In Print modes the per-element output is on stdout — the panic message
    // intentionally drops the examples block. In non-Print modes the block
    // must be present.
    let is_print_mode = std::env::var("CUBE_TEST_MODE")
        .map(|v| v.to_lowercase().starts_with("print"))
        .unwrap_or(false);
    if !is_print_mode {
        assert!(
            reason.contains("First mismatches:"),
            "missing examples header, got: {reason}"
        );
    }
}

#[test]
fn assert_equals_approx_in_slice_accepts_tensor_filter() {
    use cubek_test_utils::DimFilter;

    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Same setup as `arange_handle_row_major_slice` — the second row of
    // `actual` differs (9, 9, 9) from the expected arange (3, 4, 5).
    let actual = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 9., 9., 9.])
        .f32_host_data();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    // Equivalent to `vec![0..1, 0..3]` — only the first row is compared, so
    // the differing second row is ignored and the result is `Pass`.
    let filter = vec![DimFilter::Exact(0), DimFilter::Range { start: 0, end: 2 }];
    assert_equals_approx_in_slice(&actual, &expected, 0.001, filter)
        .as_test_outcome()
        .enforce();
}

#[test]
fn builder_matches_constructor() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Builder form: defaults to f32 + RowMajor; finalizer picks the data kind.
    let from_builder = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();

    // Equivalent construction with the long-form `TestInput::new`.
    let from_new = TestInput::new(
        client.clone(),
        shape![2, 3],
        f32::as_type_native_unchecked().storage_type(),
        StrideSpec::RowMajor,
        DataKind::Arange { scale: None },
    )
    .f32_host_data();

    assert_equals_approx(&from_builder, &from_new, 0.0)
        .as_test_outcome()
        .enforce();
}

#[test]
fn builder_overrides_stride_and_dtype() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Builder with explicit stride override should match the constructor.
    let from_builder = TestInput::builder(client.clone(), shape![2, 3])
        .stride(StrideSpec::ColMajor)
        .arange()
        .f32_host_data();

    let from_new = TestInput::new(
        client.clone(),
        shape![2, 3],
        f32::as_type_native_unchecked().storage_type(),
        StrideSpec::ColMajor,
        DataKind::Arange { scale: None },
    )
    .f32_host_data();

    assert_equals_approx(&from_builder, &from_new, 0.0)
        .as_test_outcome()
        .enforce();
}

#[test]
fn builder_linspace_produces_evenly_spaced_values() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

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
    let client = <TestRuntime as Runtime>::client(&Default::default());

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
fn host_data_typed_accessors_and_iter() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // arange tensor — values 0..6 in row-major order.
    let host = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();

    // Panicking and falling-back accessors agree on F32 data.
    assert_eq!(host.get_f32(&[1, 2]), 5.0);
    assert_eq!(host.try_get_f32(&[1, 2]), Some(5.0));
    // Wrong dtype → None instead of panic.
    assert_eq!(host.try_get_i32(&[1, 2]), None);
    assert_eq!(host.try_get_bool(&[1, 2]), None);

    // Indexed iterator walks every cell row-major.
    let collected: Vec<(Vec<usize>, f32)> = host.iter_indexed_f32().collect();
    assert_eq!(collected.len(), 6);
    assert_eq!(collected[0], (vec![0, 0], 0.0));
    assert_eq!(collected[3], (vec![1, 0], 3.0));
    assert_eq!(collected.last().unwrap(), &(vec![1, 2], 5.0));
}

#[test]
fn host_data_iter_respects_strides() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // Same logical 2x3 tensor, but stored col-major. The iterator must walk
    // the *logical* row-major order while resolving each cell through the
    // strides — so the values must match a row-major arange.
    let actual = TestInput::builder(client.clone(), shape![2, 3])
        .stride(StrideSpec::ColMajor)
        .arange()
        .f32_host_data();

    let collected: Vec<f32> = actual.iter_indexed_f32().map(|(_, v)| v).collect();
    assert_eq!(collected, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// Run with `--nocapture` to see print
#[test]
fn playground_partial_mismatch() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let eps = 0.001f32;

    let expected = TestInput::builder(client.clone(), shape![4, 4])
        .custom(vec![
            0.10, 0.20, 0.30, 0.40, 1.00, 1.10, 1.20, 1.30, 2.00, 2.10, 2.20, 2.30, 3.00, 3.10,
            3.20, 3.30,
        ])
        .f32_host_data();

    let actual = TestInput::builder(client.clone(), shape![4, 4])
        .custom(vec![
            0.10,
            0.20,
            0.30,
            0.40,
            1.0001,
            1.0999,
            1.2001,
            1.2999,
            2.01,
            2.10,
            2.21,
            2.31,
            3.50,
            3.60,
            3.70,
            f32::NAN,
        ])
        .f32_host_data();

    let result = assert_equals_approx(&actual, &expected, eps);
    assert!(
        matches!(result, ValidationResult::Fail(_)),
        "expected partial mismatch to be flagged as Fail, got {result:?}"
    );
}

#[test]
fn print_tensors_skips_shape_mismatch() {
    use cubek_test_utils::print_tensors;

    let client = <TestRuntime as Runtime>::client(&Default::default());

    let a = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();
    let b = TestInput::builder(client.clone(), shape![3, 2])
        .arange()
        .f32_host_data();

    // Shapes differ (2,3) vs (3,2) — must not panic and must not print.
    // (We can't observe stdout from here, so this is a smoke test.)
    print_tensors("mismatched", &[&a, &b], Some(0.001));
}

#[test]
fn print_tensors_skips_rank_mismatch() {
    use cubek_test_utils::print_tensors;

    let client = <TestRuntime as Runtime>::client(&Default::default());

    let r2 = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();
    let r3 = TestInput::builder(client.clone(), shape![2, 2, 3])
        .arange()
        .f32_host_data();

    // Different ranks — must not panic and must not print.
    print_tensors("mismatched_rank", &[&r2, &r3], Some(0.001));
}

#[test]
fn print_tensor_is_no_op_in_correct_mode() {
    // Smoke check: `print_tensor` must never panic regardless of the active
    // test mode, so it is safe to leave in place once a test is debugged.
    // (We can't observe stdout in a parallel test runner, so we just exercise
    // the call path on a few ranks.)
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let r2 = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();
    print_tensor("rank-2 arange", &r2);

    let r3 = TestInput::builder(client.clone(), shape![2, 2, 3])
        .arange()
        .f32_host_data();
    print_tensor("rank-3 arange", &r3);
}

#[test]
fn pretty_print_handles_rank_3() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    // 2 × 2 × 3 tensor — should print as two labeled 2×3 tables.
    let host = TestInput::builder(client.clone(), shape![2, 2, 3])
        .arange()
        .f32_host_data();

    let printed = host.pretty_print();
    assert!(
        printed.contains("[0, *, *]"),
        "missing first slice label:\n{printed}"
    );
    assert!(
        printed.contains("[1, *, *]"),
        "missing second slice label:\n{printed}"
    );
    // arange produces 0..12; the first slice spans 0..6, the second 6..12.
    assert!(
        printed.contains("0.000"),
        "expected 0.000 in printed output:\n{printed}"
    );
    assert!(
        printed.contains("11.000"),
        "expected 11.000 in printed output:\n{printed}"
    );
}

#[test]
fn pretty_print_slice_filters_rows_and_cols() {
    use cubek_test_utils::DimFilter;

    let client = <TestRuntime as Runtime>::client(&Default::default());

    // 2 × 4 arange — values 0..8 in row-major order.
    let host = TestInput::builder(client.clone(), shape![2, 4])
        .arange()
        .f32_host_data();

    // Pin row 1 only, cols 1-2 only — should print just `[1, 1]=5` and
    // `[1, 2]=6` and skip every other cell.
    let printed = host.pretty_print_slice(vec![
        DimFilter::Exact(1),
        DimFilter::Range { start: 1, end: 2 },
    ]);

    assert!(printed.contains("5.000"), "expected 5.000 in: {printed}");
    assert!(printed.contains("6.000"), "expected 6.000 in: {printed}");
    // Column 0 (value 4) and column 3 (value 7) should be filtered out.
    assert!(
        !printed.contains("4.000"),
        "should not contain 4.000: {printed}"
    );
    assert!(
        !printed.contains("7.000"),
        "should not contain 7.000: {printed}"
    );
    // Row 0 (values 0..3) should be filtered out entirely.
    assert!(
        !printed.contains("0.000"),
        "should not contain 0.000: {printed}"
    );
}

#[test]
fn pretty_print_slice_filters_leading_dims() {
    use cubek_test_utils::DimFilter;

    let client = <TestRuntime as Runtime>::client(&Default::default());

    // 3 × 2 × 2 — exercise filtering on the leading dim.
    let host = TestInput::builder(client.clone(), shape![3, 2, 2])
        .arange()
        .f32_host_data();

    // Pin leading dim to 1; the row/col axes always use `Any`.
    let filter = vec![DimFilter::Exact(1), DimFilter::Any, DimFilter::Any];
    let printed = host.pretty_print_slice(filter);

    assert!(
        printed.contains("[1, *, *]"),
        "expected slice [1, *, *]:\n{printed}"
    );
    assert!(
        !printed.contains("[0, *, *]"),
        "should not include slice [0, *, *]:\n{printed}"
    );
    assert!(
        !printed.contains("[2, *, *]"),
        "should not include slice [2, *, *]:\n{printed}"
    );
}

#[test]
fn builder_uniform_values_in_range() {
    // Range/property check rather than seed-stability — see the note on the
    // normal test above.
    let client = <TestRuntime as Runtime>::client(&Default::default());

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
