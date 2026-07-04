use std::fmt::Display;

use cubecl::features::TypeUsage;
use cubecl::{TestRuntime, prelude::*};
use cubecl::std::tensor::TensorHandle;

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

pub(crate) fn tensorhandler_from_data<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    shape: Vec<usize>,
    data: &[F],
    dtype: cubecl::ir::Type,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(F::as_bytes(data));
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    TensorHandle::new(handle, shape, strides, dtype)
}

pub(crate) fn tensorhandler_from_data_col_major<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    shape: Vec<usize>,
    data: &[F],
    dtype: cubecl::ir::Type,
) -> TensorHandle<R> {
    let handle = client.create_from_slice(F::as_bytes(data));
    let mut strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    TensorHandle::new(handle, shape, strides, dtype)
}

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<F: Float + CubeElement + Display>(
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual_bytes = client.read_one(out.handle.clone()).unwrap();
    let actual = F::from_bytes(&actual_bytes);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        let allowed_error = (epsilon * e.to_f32().unwrap().abs()).max(epsilon);

        if f32::is_nan(a.to_f32().unwrap())
            || f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()) >= allowed_error
        {
            return Err(format!(
                "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                i,
                *a,
                *e,
                f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()),
                epsilon
            ));
        }
    }

    Ok(())
}
