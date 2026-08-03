use cubecl::features::TypeUsage;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::{Shape, Strides};
use cubecl::{TestRuntime, prelude::*};
use cubek_test_utils::{HostData, HostDataVec, StridedLayout, TestInput, ValidationResult};

/// Returns `true` (and prints a skip notice) when the active backend can't do
/// reliable arithmetic in `F`, so the precision-sensitive QR tests would fail
/// for a reason unrelated to the algorithm. Tests call this first and return
/// early to avoid such false failures; CUDA (full f64) runs them all.
///
/// Two signals are combined:
///  - The backend's advertised type support (`supported_uses`), which catches
///    backends that honestly report a missing type.
///  - A WGSL special case: the WGSL spec has **no 64-bit float**, yet the
///    `wgpu<wgsl>` backend still advertises full f64 support and then silently
///    produces garbage. We can't trust the feature flag there, so f64 on any
///    WGSL runtime is treated as unsupported by name.
pub(crate) fn dtype_unsupported<F: Float + CubeElement>(
    client: &ComputeClient<TestRuntime>,
) -> bool {
    let unsupported_by_features = !F::supported_uses(client).contains(TypeUsage::Arithmetic);

    // f64 is 8 bytes; f32/f16/bf16 are smaller. The QR test suite only uses
    // f32 and f64, so size == 8 uniquely identifies f64.
    let is_f64 = core::mem::size_of::<F>() == 8;
    let runtime = TestRuntime::name(client);
    let wgsl_f64 = is_f64 && runtime.contains("wgsl");

    if unsupported_by_features || wgsl_f64 {
        println!(
            "Skipping: backend `{runtime}` does not reliably support `{}` arithmetic.",
            core::any::type_name::<F>()
        );
        true
    } else {
        false
    }
}

/// Build a device tensor from logical row-major values via the shared
/// `TestInput` builder (which owns the stride math). The builder's payload
/// type is f32; the QR test matrices hold small integers, so the round-trip
/// is lossless for both f32 and f64.
fn device_input<F: Float + CubeElement>(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    data_row_major: &[F],
    layout: StridedLayout,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::from(shape))
        .dtype(F::as_type_native_unchecked().storage_type())
        .layout(layout)
        .custom(data_row_major.iter().map(|v| v.to_f32().unwrap()).collect())
        .generate_without_host_data()
}

pub(crate) fn col_major_input<F: Float + CubeElement>(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    data_row_major: &[F],
) -> TensorHandle<TestRuntime> {
    // ColMajor requires rank >= 2; for vectors both layouts are the same.
    let layout = if shape.len() >= 2 {
        StridedLayout::ColMajor
    } else {
        StridedLayout::RowMajor
    };
    device_input(client, shape, data_row_major, layout)
}

pub(crate) fn row_major_input<F: Float + CubeElement>(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    data_row_major: &[F],
) -> TensorHandle<TestRuntime> {
    device_input(client, shape, data_row_major, StridedLayout::RowMajor)
}

/// Wrap logical row-major host values in a `HostData` blob (f64 payload for
/// f64 tensors, f32 otherwise) for the shared comparator.
fn host_data<F: Float + CubeElement>(shape: Vec<usize>, values: &[F]) -> HostData {
    let storage = F::as_type_native_unchecked().storage_type();
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let values: Vec<f64> = values.iter().map(|v| v.to_f64().unwrap()).collect();
    HostData {
        data: HostDataVec::from((values, storage)),
        shape: Shape::from(shape),
        strides: Strides::from(strides),
    }
}

/// Compare two logical row-major value slices through cubek-test-utils'
/// shared comparator (relative epsilon with an absolute floor), panicking
/// with its mismatch report on failure.
pub(crate) fn assert_equals_approx<F: Float + CubeElement>(
    actual: &[F],
    expected: &[F],
    shape: Vec<usize>,
    epsilon: f32,
) {
    let actual = host_data::<F>(shape.clone(), actual);
    let expected = host_data::<F>(shape, expected);
    match cubek_test_utils::assert_equals_approx(&actual, &expected, epsilon) {
        ValidationResult::Pass => {}
        ValidationResult::Fail(msg) | ValidationResult::Error(msg) => panic!("{msg}"),
        ValidationResult::Skipped(msg) => panic!("unexpected validation skip: {msg}"),
    }
}
