use cubecl::zspace::shape;
use cubek_test_utils::{TestInput, ValidationResult, assert_equals_approx, print_tensor};

/// Run with `--nocapture` to see print
#[test]
fn playground_partial_mismatch() {
    let client = cubecl::test_device().client();
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

    let client = cubecl::test_device().client();

    let a = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();
    let b = TestInput::builder(client.clone(), shape![3, 2])
        .arange()
        .f32_host_data();

    // Shapes differ (2,3) vs (3,2): must not panic and must not print.
    // (We can't observe stdout from here, so this is a smoke test.)
    print_tensors("mismatched", &[&a, &b], Some(0.001));
}

#[test]
fn print_tensors_skips_rank_mismatch() {
    use cubek_test_utils::print_tensors;

    let client = cubecl::test_device().client();

    let r2 = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();
    let r3 = TestInput::builder(client.clone(), shape![2, 2, 3])
        .arange()
        .f32_host_data();

    // Different ranks: must not panic and must not print.
    print_tensors("mismatched_rank", &[&r2, &r3], Some(0.001));
}

#[test]
fn print_tensor_is_no_op_in_correct_mode() {
    // Smoke check: `print_tensor` must never panic regardless of the active
    // test mode, so it is safe to leave in place once a test is debugged.
    // (We can't observe stdout in a parallel test runner, so we just exercise
    // the call path on a few ranks.)
    let client = cubecl::test_device().client();

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
    let client = cubecl::test_device().client();

    // 2 × 2 × 3 tensor: should print as two labeled 2×3 tables.
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

    let client = cubecl::test_device().client();

    // 2 × 4 arange: values 0..8 in row-major order.
    let host = TestInput::builder(client.clone(), shape![2, 4])
        .arange()
        .f32_host_data();

    // Pin row 1 only, cols 1-2 only: should print just `[1, 1]=5` and
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

    let client = cubecl::test_device().client();

    // 3 × 2 × 2: exercise filtering on the leading dim.
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
