use cubecl::zspace::shape;
use cubek_test_utils::{
    TestInput, ValidationResult, assert_equals_approx, assert_equals_approx_in_slice,
};

#[test]
fn fail_message_contains_aggregate_stats_and_examples() {
    let client = cubecl::test_device().client();

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

    // In Print modes the per-element output is on stdout: the panic message
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

    let client = cubecl::test_device().client();

    // Same setup as `arange_handle_row_major_slice`: the second row of
    // `actual` differs (9, 9, 9) from the expected arange (3, 4, 5).
    let actual = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 9., 9., 9.])
        .f32_host_data();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    // Equivalent to `vec![0..1, 0..3]`: only the first row is compared, so
    // the differing second row is ignored and the result is `Pass`.
    let filter = vec![DimFilter::Exact(0), DimFilter::Range { start: 0, end: 2 }];
    assert_equals_approx_in_slice(&actual, &expected, 0.001, filter)
        .as_test_outcome()
        .enforce();
}
