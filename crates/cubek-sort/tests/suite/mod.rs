use cubecl::TestRuntime;
use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
use cubecl::prelude::*;
use cubek_sort::{SortKey, SortOrder, sort_keys};
use half::{bf16, f16};
use rand::{Rng, SeedableRng, rngs::StdRng};

use SortOrder::{Ascending, Descending};

fn random_values<T>(size: usize, seed: u64) -> Vec<T>
where
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.random()).collect()
}

fn is_supported(client: &ComputeClient<TestRuntime>, types: &[ElemType]) -> bool {
    types
        .iter()
        .all(|t| client.properties().features.supports_type(*t))
}

fn run_sort_test<T>(input: &[T], order: SortOrder, required_types: &[ElemType])
where
    T: SortKey + CubeElement + Clone + PartialOrd + std::fmt::Debug,
    T::Radix: SortKey<Radix = T::Radix>,
{
    let client = TestRuntime::client(&Default::default());

    if !is_supported(&client, required_types) {
        return;
    }

    let num_items = input.len();
    if num_items == 0 {
        return;
    }

    let input_handle = client.create_from_slice(T::as_bytes(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    let shape = [num_items];
    let strides = [1];

    let input_ref =
        unsafe { TensorHandleRef::from_raw_parts(&input_handle, &strides, &shape, size_of::<T>()) };
    let output_ref = unsafe {
        TensorHandleRef::from_raw_parts(&output_handle, &strides, &shape, size_of::<T>())
    };

    let result = sort_keys::<TestRuntime, T>(&client, input_ref, output_ref, num_items, order);
    result.expect("Sort failed");

    let bytes = client.read_one(output_handle);
    let output = T::from_bytes(&bytes);

    let mut expected = input.to_vec();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if order.is_descending() {
        expected.reverse();
    }

    assert_eq!(output.len(), expected.len(), "Length mismatch");
    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, exp,
            "Mismatch at index {i}: got {got:?}, expected {exp:?}"
        );
    }
}

// f32 tests (comprehensive)

#[test]
fn test_f32_empty() {
    run_sort_test::<f32>(&[], Ascending, &[]);
}

#[test]
fn test_f32_single() {
    run_sort_test(&[3.5f32], Ascending, &[]);
}

#[test]
fn test_f32_two_elements() {
    run_sort_test(&[1.0f32, -1.0], Ascending, &[]);
    run_sort_test(&[-1.0f32, 1.0], Ascending, &[]);
}

#[test]
fn test_f32_small_random() {
    run_sort_test(&random_values::<f32>(64, 12345), Ascending, &[]);
}

#[test]
fn test_f32_medium_random() {
    run_sort_test(&random_values::<f32>(1009, 54321), Ascending, &[]);
}

#[test]
fn test_f32_large_random() {
    run_sort_test(&random_values::<f32>(65537, 99999), Ascending, &[]);
}

#[test]
fn test_f32_already_sorted() {
    let values: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    run_sort_test(&values, Ascending, &[]);
}

#[test]
fn test_f32_reverse_sorted() {
    let values: Vec<f32> = (0..1024).rev().map(|i| i as f32).collect();
    run_sort_test(&values, Ascending, &[]);
}

#[test]
fn test_f32_duplicates() {
    let values: Vec<f32> = (0..1024).map(|i| (i % 16) as f32 - 8.0).collect();
    run_sort_test(&values, Ascending, &[]);
}

#[test]
fn test_f32_all_same() {
    run_sort_test(&vec![42.0f32; 1024], Ascending, &[]);
}

#[test]
fn test_f32_infinities() {
    run_sort_test(
        &[f32::NEG_INFINITY, -1.0, 0.0, 1.0, f32::INFINITY],
        Ascending,
        &[],
    );
    run_sort_test(
        &[
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            0.0,
            f32::NEG_INFINITY,
        ],
        Ascending,
        &[],
    );
}

#[test]
fn test_f32_zero_variants() {
    run_sort_test(&[0.0f32, -0.0, 0.0, -0.0], Ascending, &[]);
}

#[test]
fn test_f32_subnormals() {
    let subnormal_pos = f32::from_bits(0x0000_0001);
    let subnormal_neg = f32::from_bits(0x8000_0001);
    run_sort_test(
        &[0.0, subnormal_pos, -0.0, subnormal_neg, f32::MIN_POSITIVE],
        Ascending,
        &[],
    );
}

#[test]
fn test_f32_special_mixed() {
    let mut values = random_values::<f32>(100, 11111);
    values.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MAX,
        -f32::MAX,
    ]);
    run_sort_test(&values, Ascending, &[]);
}

#[test]
fn test_f32_descending() {
    run_sort_test(&random_values::<f32>(1009, 12345), Descending, &[]);
}

#[test]
fn test_f32_descending_with_infinities() {
    run_sort_test(
        &[f32::NEG_INFINITY, -1.0, 0.0, 1.0, f32::INFINITY],
        Descending,
        &[],
    );
}

// u32 tests (spot checks)

#[test]
fn test_u32_basic() {
    run_sort_test(&[42u32], Ascending, &[]);
    run_sort_test(&[2u32, 1], Ascending, &[]);
    run_sort_test(&random_values::<u32>(1009, 12345), Ascending, &[]);
}

#[test]
fn test_u32_extremes() {
    run_sort_test(&[u32::MIN, u32::MAX, 0, 1, u32::MAX - 1], Ascending, &[]);
}

#[test]
fn test_u32_descending() {
    run_sort_test(&random_values::<u32>(1009, 12345), Descending, &[]);
}

// i32 tests (spot checks)

#[test]
fn test_i32_basic() {
    run_sort_test(&[-42i32], Ascending, &[]);
    run_sort_test(&[1i32, -1], Ascending, &[]);
    run_sort_test(&random_values::<i32>(1009, 12345), Ascending, &[]);
}

#[test]
fn test_i32_sign_boundary() {
    run_sort_test(&[i32::MIN, i32::MAX, 0, -1, 1], Ascending, &[]);
    let values: Vec<i32> = (-500..500).collect();
    run_sort_test(&values, Ascending, &[]);
}

#[test]
fn test_i32_descending() {
    run_sort_test(&random_values::<i32>(1009, 12345), Descending, &[]);
}

// 16-bit types

const U16: ElemType = ElemType::UInt(UIntKind::U16);
const I16: ElemType = ElemType::Int(IntKind::I16);
const F16: ElemType = ElemType::Float(FloatKind::F16);
const BF16: ElemType = ElemType::Float(FloatKind::BF16);

#[test]
fn test_u16_basic() {
    run_sort_test(&[42u16], Ascending, &[U16]);
    run_sort_test(&random_values::<u16>(1009, 12345), Ascending, &[U16]);
    run_sort_test(&[u16::MIN, u16::MAX, 1, 100], Ascending, &[U16]);
}

#[test]
fn test_i16_basic() {
    run_sort_test(&[-42i16], Ascending, &[I16]);
    run_sort_test(&random_values::<i16>(1009, 12345), Ascending, &[I16]);
    run_sort_test(&[i16::MIN, i16::MAX, 0, -1, 1], Ascending, &[I16]);
}

fn random_f16(size: usize, seed: u64) -> Vec<f16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| f16::from_f32((rng.random::<f32>() - 0.5) * 1000.0))
        .collect()
}

fn random_bf16(size: usize, seed: u64) -> Vec<bf16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| bf16::from_f32((rng.random::<f32>() - 0.5) * 1000.0))
        .collect()
}

#[test]
fn test_f16_basic() {
    run_sort_test(&[f16::from_f32(3.5)], Ascending, &[F16, U16]);
    run_sort_test(&random_f16(1009, 12345), Ascending, &[F16, U16]);
}

#[test]
fn test_f16_special() {
    run_sort_test(
        &[
            f16::NEG_INFINITY,
            f16::from_f32(-1.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::INFINITY,
        ],
        Ascending,
        &[F16, U16],
    );
}

#[test]
fn test_bf16_basic() {
    run_sort_test(&[bf16::from_f32(3.5)], Ascending, &[BF16, U16]);
    run_sort_test(&random_bf16(1009, 12345), Ascending, &[BF16, U16]);
}

#[test]
fn test_bf16_special() {
    run_sort_test(
        &[
            bf16::NEG_INFINITY,
            bf16::from_f32(-1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::INFINITY,
        ],
        Ascending,
        &[BF16, U16],
    );
}

// 8-bit types

const U8: ElemType = ElemType::UInt(UIntKind::U8);
const I8: ElemType = ElemType::Int(IntKind::I8);

#[test]
fn test_u8_basic() {
    run_sort_test(&[42u8], Ascending, &[U8]);
    run_sort_test(&random_values::<u8>(1009, 12345), Ascending, &[U8]);
    run_sort_test(&[u8::MIN, u8::MAX, 1, 100], Ascending, &[U8]);
}

#[test]
fn test_u8_all_values() {
    let values: Vec<u8> = (0..=255).collect();
    run_sort_test(&values, Ascending, &[U8]);
}

#[test]
fn test_i8_basic() {
    run_sort_test(&[-42i8], Ascending, &[I8]);
    run_sort_test(&random_values::<i8>(1009, 12345), Ascending, &[I8]);
    run_sort_test(&[i8::MIN, i8::MAX, 0, -1, 1], Ascending, &[I8]);
}

#[test]
fn test_i8_all_values() {
    let values: Vec<i8> = (-128..=127).map(|x| x as i8).collect();
    run_sort_test(&values, Ascending, &[I8]);
}

// 64-bit types

const U64: ElemType = ElemType::UInt(UIntKind::U64);
const I64: ElemType = ElemType::Int(IntKind::I64);
const F64: ElemType = ElemType::Float(FloatKind::F64);

#[test]
fn test_u64_basic() {
    run_sort_test(&[42u64], Ascending, &[U64]);
    run_sort_test(&random_values::<u64>(1009, 12345), Ascending, &[U64]);
    run_sort_test(&[u64::MIN, u64::MAX, 1, u64::MAX / 2], Ascending, &[U64]);
}

#[test]
fn test_i64_basic() {
    run_sort_test(&[-42i64], Ascending, &[I64]);
    run_sort_test(&random_values::<i64>(1009, 12345), Ascending, &[I64]);
    run_sort_test(&[i64::MIN, i64::MAX, 0, -1, 1], Ascending, &[I64]);
}

#[test]
fn test_f64_basic() {
    run_sort_test(&[3.5f64], Ascending, &[F64, U64]);
    run_sort_test(&random_values::<f64>(1009, 12345), Ascending, &[F64, U64]);
}

#[test]
fn test_f64_special() {
    run_sort_test(
        &[f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY],
        Ascending,
        &[F64, U64],
    );
    run_sort_test(&[0.0f64, -0.0, 0.0, -0.0], Ascending, &[F64, U64]);
}

/// Test sorting 150M elements to catch out-of-bounds memory access issues.
#[test]
fn test_large_scale_u32() {
    const SIZE: usize = 150 * 1024 * 1024;

    let client = TestRuntime::client(&Default::default());
    let data: Vec<u32> = (0..SIZE as u32).rev().collect();

    let input_handle = client.create_from_slice(u32::as_bytes(&data));
    let output_handle = client.empty(SIZE * std::mem::size_of::<u32>());

    let shape = [SIZE];
    let strides = [1];

    let input_ref = unsafe {
        TensorHandleRef::from_raw_parts(&input_handle, &strides, &shape, size_of::<u32>())
    };
    let output_ref = unsafe {
        TensorHandleRef::from_raw_parts(&output_handle, &strides, &shape, size_of::<u32>())
    };

    sort_keys::<TestRuntime, u32>(&client, input_ref, output_ref, SIZE, Ascending)
        .expect("Sort failed");

    // Verify sortedness by checking consecutive samples across the entire range
    let bytes = client.read_one(output_handle);
    let output = u32::from_bytes(&bytes);
    assert_eq!(output.len(), SIZE);
    let mut prev = output[0];
    for i in (1..SIZE).step_by(10_000) {
        let curr = output[i];
        assert!(
            prev <= curr,
            "Not sorted at index {}: {} > {}",
            i,
            prev,
            curr
        );
        prev = curr;
    }
}
