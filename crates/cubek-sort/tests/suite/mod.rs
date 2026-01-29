//! End-to-end tests for cubek-sort.
//!
//! These tests verify that the sorting algorithm produces correct results
//! on actual GPU backends.

use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubek_sort::sort_keys;
use rand::{Rng, SeedableRng, rngs::StdRng};

// =============================================================================
// Test runner for i32
// =============================================================================

/// Run a sort test with i32 input and verify correctness.
fn run_sort_test_i32(input: &[i32]) {
    let client = TestRuntime::client(&Default::default());
    let num_items = input.len();

    if num_items == 0 {
        return;
    }

    // Create input and output handles
    let input_handle = client.create_from_slice(i32::as_bytes(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    let shape = [num_items];
    let strides = [1];

    let input_ref = unsafe {
        TensorHandleRef::from_raw_parts(&input_handle, &strides, &shape, size_of::<i32>())
    };
    let output_ref = unsafe {
        TensorHandleRef::from_raw_parts(&output_handle, &strides, &shape, size_of::<i32>())
    };

    // Run the sort with default strategy
    let result = sort_keys::<TestRuntime, i32>(&client, input_ref, output_ref, num_items, None);

    match result {
        Ok(()) => {}
        Err(e) => panic!("Sort failed: {e:?}"),
    }

    // Read back the results
    let bytes = client.read_one(output_handle);
    let output = i32::from_bytes(&bytes);

    // Compute expected result using CPU sort
    let mut expected = input.to_vec();
    expected.sort();

    // Compare
    assert_eq!(
        output.len(),
        expected.len(),
        "Output length mismatch: got {}, expected {}",
        output.len(),
        expected.len()
    );

    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, exp,
            "Mismatch at index {i}: got {got}, expected {exp}\nFull output: {output:?}\nExpected: {expected:?}"
        );
    }
}

// =============================================================================
// Test runner for f32
// =============================================================================

/// Run a sort test with f32 input and verify correctness.
/// NaN values are sorted by their bitwise representation (matching CUB behavior).
fn run_sort_test_f32(input: &[f32]) {
    let client = TestRuntime::client(&Default::default());
    let num_items = input.len();

    if num_items == 0 {
        return;
    }

    // Create input and output handles
    let input_handle = client.create_from_slice(f32::as_bytes(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    let shape = [num_items];
    let strides = [1];

    let input_ref = unsafe {
        TensorHandleRef::from_raw_parts(&input_handle, &strides, &shape, size_of::<f32>())
    };
    let output_ref = unsafe {
        TensorHandleRef::from_raw_parts(&output_handle, &strides, &shape, size_of::<f32>())
    };

    // Run the sort with default strategy
    let result = sort_keys::<TestRuntime, f32>(&client, input_ref, output_ref, num_items, None);

    match result {
        Ok(()) => {}
        Err(e) => panic!("Sort failed: {e:?}"),
    }

    // Read back the results
    let bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&bytes);

    // Compute expected result using CPU sort with same ordering as GPU
    // For f32: -inf < negative < -0 < +0 < positive < +inf < NaN (by bits)
    let mut expected = input.to_vec();
    expected.sort_by(|a, b| f32_sort_order(*a, *b));

    // Compare
    assert_eq!(
        output.len(),
        expected.len(),
        "Output length mismatch: got {}, expected {}",
        output.len(),
        expected.len()
    );

    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        // Use bitwise comparison to handle NaN correctly
        let got_bits = got.to_bits();
        let exp_bits = exp.to_bits();
        assert_eq!(
            got_bits, exp_bits,
            "Mismatch at index {i}: got {got} (bits: {got_bits:#x}), expected {exp} (bits: {exp_bits:#x})"
        );
    }
}

/// Sort ordering for f32 that matches the GPU radix sort behavior.
/// This uses the same bit transformation as SortKey::to_radix_key.
fn f32_sort_order(a: f32, b: f32) -> std::cmp::Ordering {
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();

    // Transform to radix key (same as SortKey impl):
    // Positive floats: flip sign bit (0x8000_0000 ^ bits)
    // Negative floats: flip all bits (!bits)
    let a_key = if (a_bits as i32) < 0 {
        !a_bits
    } else {
        a_bits ^ 0x8000_0000
    };
    let b_key = if (b_bits as i32) < 0 {
        !b_bits
    } else {
        b_bits ^ 0x8000_0000
    };

    a_key.cmp(&b_key)
}

/// Run a sort test with the given input and verify correctness.
fn run_sort_test_u32(input: &[u32]) {
    let client = TestRuntime::client(&Default::default());
    let num_items = input.len();

    if num_items == 0 {
        return;
    }

    // Create input and output handles
    let input_handle = client.create_from_slice(u32::as_bytes(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    let shape = [num_items];
    let strides = [1];

    let input_ref = unsafe {
        TensorHandleRef::from_raw_parts(&input_handle, &strides, &shape, size_of::<u32>())
    };
    let output_ref = unsafe {
        TensorHandleRef::from_raw_parts(&output_handle, &strides, &shape, size_of::<u32>())
    };

    // Run the sort with default strategy
    let result = sort_keys::<TestRuntime, u32>(&client, input_ref, output_ref, num_items, None);

    match result {
        Ok(()) => {}
        Err(e) => panic!("Sort failed: {e:?}"),
    }

    // Read back the results
    let bytes = client.read_one(output_handle);
    let output = u32::from_bytes(&bytes);

    // Compute expected result using CPU sort
    let mut expected = input.to_vec();
    expected.sort();

    // Compare
    assert_eq!(
        output.len(),
        expected.len(),
        "Output length mismatch: got {}, expected {}",
        output.len(),
        expected.len()
    );

    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, exp,
            "Mismatch at index {i}: got {got}, expected {exp}\nFull output: {output:?}\nExpected: {expected:?}"
        );
    }
}

/// Generate random u32 values.
fn random_u32_values(size: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.random()).collect()
}

/// Generate sequential values in reverse order (worst case for some sorts).
fn reverse_sequential(size: usize) -> Vec<u32> {
    (0..size as u32).rev().collect()
}

/// Generate values that are already sorted.
fn already_sorted(size: usize) -> Vec<u32> {
    (0..size as u32).collect()
}

/// Generate values with many duplicates.
fn many_duplicates(size: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Only use 16 distinct values
    (0..size).map(|_| rng.random::<u32>() % 16).collect()
}

// =============================================================================
// Test cases
// =============================================================================

#[test]
fn test_empty() {
    run_sort_test_u32(&[]);
}

#[test]
fn test_single_element() {
    run_sort_test_u32(&[42]);
}

#[test]
fn test_two_elements_sorted() {
    run_sort_test_u32(&[1, 2]);
}

#[test]
fn test_two_elements_reversed() {
    run_sort_test_u32(&[2, 1]);
}

#[test]
fn test_small_random() {
    run_sort_test_u32(&random_u32_values(16, 12345));
}

#[test]
fn test_medium_random() {
    run_sort_test_u32(&random_u32_values(256, 12345));
}

#[test]
fn test_large_random() {
    run_sort_test_u32(&random_u32_values(4096, 12345));
}

#[test]
fn test_xlarge_random() {
    run_sort_test_u32(&random_u32_values(65536, 12345));
}

#[test]
fn test_already_sorted_small() {
    run_sort_test_u32(&already_sorted(64));
}

#[test]
fn test_already_sorted_large() {
    run_sort_test_u32(&already_sorted(4096));
}

#[test]
fn test_reverse_sorted_small() {
    run_sort_test_u32(&reverse_sequential(64));
}

#[test]
fn test_reverse_sorted_large() {
    run_sort_test_u32(&reverse_sequential(4096));
}

#[test]
fn test_duplicates_small() {
    run_sort_test_u32(&many_duplicates(64, 12345));
}

#[test]
fn test_duplicates_large() {
    run_sort_test_u32(&many_duplicates(4096, 12345));
}

#[test]
fn test_all_same_value() {
    run_sort_test_u32(&vec![42u32; 1024]);
}

#[test]
fn test_all_zeros() {
    run_sort_test_u32(&vec![0u32; 1024]);
}

#[test]
fn test_all_max() {
    run_sort_test_u32(&vec![u32::MAX; 1024]);
}

#[test]
fn test_min_and_max() {
    let mut values = vec![u32::MAX; 512];
    values.extend(vec![0u32; 512]);
    run_sort_test_u32(&values);
}

#[test]
fn test_power_of_two_size() {
    run_sort_test_u32(&random_u32_values(1024, 54321));
}

#[test]
fn test_non_power_of_two_size() {
    run_sort_test_u32(&random_u32_values(1000, 54321));
}

#[test]
fn test_prime_size() {
    run_sort_test_u32(&random_u32_values(1009, 54321)); // 1009 is prime
}

#[test]
fn test_odd_size() {
    run_sort_test_u32(&random_u32_values(1023, 54321));
}

// =============================================================================
// i32 tests
// =============================================================================

/// Generate random i32 values (full range including negatives).
fn random_i32_values(size: usize, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.random()).collect()
}

#[test]
fn test_i32_single_negative() {
    run_sort_test_i32(&[-42]);
}

#[test]
fn test_i32_two_elements() {
    run_sort_test_i32(&[1, -1]);
}

#[test]
fn test_i32_small_random() {
    run_sort_test_i32(&random_i32_values(64, 12345));
}

#[test]
fn test_i32_medium_random() {
    run_sort_test_i32(&random_i32_values(1024, 12345));
}

#[test]
fn test_i32_large_random() {
    run_sort_test_i32(&random_i32_values(8192, 12345));
}

#[test]
fn test_i32_all_negative() {
    let values: Vec<i32> = (-1000..0).collect();
    run_sort_test_i32(&values);
}

#[test]
fn test_i32_all_positive() {
    let values: Vec<i32> = (0..1000).collect();
    run_sort_test_i32(&values);
}

#[test]
fn test_i32_mixed_sign() {
    let values: Vec<i32> = (-500..500).collect();
    run_sort_test_i32(&values);
}

#[test]
fn test_i32_reverse_mixed() {
    let values: Vec<i32> = (-500..500).rev().collect();
    run_sort_test_i32(&values);
}

#[test]
fn test_i32_extremes() {
    // Test with i32::MIN and i32::MAX
    run_sort_test_i32(&[i32::MIN, i32::MAX, 0, -1, 1]);
}

#[test]
fn test_i32_around_zero() {
    // Values clustered around zero
    run_sort_test_i32(&[-3, -2, -1, 0, 1, 2, 3]);
}

#[test]
fn test_i32_min_values() {
    // Multiple i32::MIN values
    run_sort_test_i32(&[i32::MIN, i32::MIN, 0, i32::MIN]);
}

#[test]
fn test_i32_max_values() {
    // Multiple i32::MAX values
    run_sort_test_i32(&[i32::MAX, i32::MAX, 0, i32::MAX]);
}

#[test]
fn test_i32_boundary_crossing() {
    // Values that cross the sign boundary in interesting ways
    let mut values = vec![i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX];
    values.extend(random_i32_values(100, 99999));
    run_sort_test_i32(&values);
}

#[test]
fn test_i32_duplicates_negative() {
    // Many duplicate negative values
    let values: Vec<i32> = (0..1024).map(|i| -(i % 8)).collect();
    run_sort_test_i32(&values);
}

// =============================================================================
// f32 tests
// =============================================================================

/// Generate random f32 values in a reasonable range.
fn random_f32_values(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| {
            // Generate in range [-1e6, 1e6] for variety
            (rng.random::<f32>() - 0.5) * 2e6
        })
        .collect()
}

#[test]
fn test_f32_single() {
    run_sort_test_f32(&[3.5]);
}

#[test]
fn test_f32_two_elements() {
    run_sort_test_f32(&[1.0, -1.0]);
}

#[test]
fn test_f32_small_random() {
    run_sort_test_f32(&random_f32_values(64, 12345));
}

#[test]
fn test_f32_medium_random() {
    run_sort_test_f32(&random_f32_values(1024, 12345));
}

#[test]
fn test_f32_large_random() {
    run_sort_test_f32(&random_f32_values(8192, 12345));
}

#[test]
fn test_f32_positive_infinity() {
    run_sort_test_f32(&[1.0, f32::INFINITY, 0.0, -1.0, f32::INFINITY]);
}

#[test]
fn test_f32_negative_infinity() {
    run_sort_test_f32(&[1.0, f32::NEG_INFINITY, 0.0, -1.0, f32::NEG_INFINITY]);
}

#[test]
fn test_f32_both_infinities() {
    run_sort_test_f32(&[
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]);
}

#[test]
fn test_f32_nan_single() {
    // NaN should sort after +inf (by bit pattern)
    run_sort_test_f32(&[1.0, f32::NAN, 0.0]);
}

#[test]
fn test_f32_nan_multiple() {
    // Multiple NaN values with different bit patterns
    let nan1 = f32::NAN;
    let nan2 = f32::from_bits(0x7FC0_0001); // Different NaN payload
    let nan3 = f32::from_bits(0x7FFF_FFFF); // Max positive NaN
    run_sort_test_f32(&[1.0, nan1, 0.0, nan2, -1.0, nan3]);
}

#[test]
fn test_f32_negative_nan() {
    // Negative NaN (sign bit set)
    let neg_nan = f32::from_bits(0xFFC0_0000);
    run_sort_test_f32(&[1.0, neg_nan, 0.0, f32::NAN, -1.0]);
}

#[test]
fn test_f32_all_special_values() {
    // Comprehensive test with all special float values
    let neg_nan = f32::from_bits(0xFFC0_0000);
    run_sort_test_f32(&[
        f32::NEG_INFINITY,
        -f32::MAX,
        -1.0,
        -f32::MIN_POSITIVE,
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        1.0,
        f32::MAX,
        f32::INFINITY,
        neg_nan,
        f32::NAN,
    ]);
}

#[test]
fn test_f32_subnormals() {
    // Subnormal (denormalized) numbers
    let subnormal_pos = f32::from_bits(0x0000_0001); // Smallest positive subnormal
    let subnormal_neg = f32::from_bits(0x8000_0001); // Smallest negative subnormal
    run_sort_test_f32(&[
        0.0,
        subnormal_pos,
        -0.0,
        subnormal_neg,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);
}

#[test]
fn test_f32_zero_variants() {
    // +0.0 and -0.0 are different bit patterns
    run_sort_test_f32(&[0.0, -0.0, 0.0, -0.0]);
}

#[test]
fn test_f32_near_zero() {
    // Values very close to zero
    run_sort_test_f32(&[1e-38, -1e-38, 1e-30, -1e-30, 0.0, -0.0]);
}

#[test]
fn test_f32_large_magnitude() {
    // Very large and very small magnitudes
    run_sort_test_f32(&[1e38, -1e38, 1e-38, -1e-38, 0.0]);
}

#[test]
fn test_f32_mixed_with_special() {
    // Mix of normal values and special values
    let mut values = random_f32_values(100, 11111);
    values.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
        f32::MAX,
        -f32::MAX,
    ]);
    run_sort_test_f32(&values);
}

#[test]
fn test_f32_duplicates() {
    // Many duplicate float values
    let values: Vec<f32> = (0..1024).map(|i| (i % 16) as f32 - 8.0).collect();
    run_sort_test_f32(&values);
}
