//! End-to-end tests for cubek-sort.
//!
//! These tests verify that the sorting algorithm produces correct results
//! on actual GPU backends.

use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubek_sort::sort_keys;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Run a sort test with the given input and verify correctness.
fn run_sort_test_u32(input: &[u32]) {
    let client = TestRuntime::client(&Default::default());
    let num_items = input.len();

    if num_items == 0 {
        return;
    }

    // Create input and output handles
    let input_handle = client.create_from_slice(u32::as_bytes(input));
    let output_handle = client.empty(num_items * size_of::<u32>());

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
