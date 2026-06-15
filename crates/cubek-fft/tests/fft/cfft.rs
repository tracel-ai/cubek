use cubecl::{
    frontend::CubePrimitive,
    prelude::StorageType,
    std::tensor::TensorHandle,
    {Runtime, TestRuntime},
};
use cubek_fft::{CfftBindings, FftMode, cfft, cfft_launch_any_size};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, TestInput, TestOutcome,
    ValidationResult, assert_equals_approx, launch_and_capture_outcome,
};

fn empty_tensor(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    let elems = shape.iter().product::<usize>();
    TensorHandle::<TestRuntime>::new_contiguous(shape, client.empty(elems * dtype.size()), dtype)
}

/// Scale every element of an f32 `HostData` by `factor`.
fn scaled(mut host: HostData, factor: f32) -> HostData {
    match &mut host.data {
        HostDataVec::F32(values) => values.iter_mut().for_each(|v| *v *= factor),
        _ => panic!("expected f32 host data"),
    }
    host
}

fn combine_re_im(re: ValidationResult, im: ValidationResult) -> ValidationResult {
    use ValidationResult::*;
    match (re, im) {
        (Fail(e), _) | (_, Fail(e)) => Fail(e),
        (Error(e), _) | (_, Error(e)) => Error(e),
        (Skipped(r1), Skipped(r2)) => Skipped(format!("{r1}, {r2}")),
        (Skipped(r), Pass) | (Pass, Skipped(r)) => Skipped(r),
        (Pass, Pass) => Pass,
    }
}

/// Forward then inverse along `dim` recovers the input scaled by `n` (the
/// transform is unnormalized in both directions).
fn cfft_roundtrip_case(signal_shape: Vec<usize>, dim: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let n = signal_shape[dim] as f32;

    let (input_re, input_re_data) = TestInput::builder(client.clone(), signal_shape.clone())
        .dtype(dtype)
        .uniform(42, -1., 1.)
        .generate_with_f32_host_data();
    let (input_im, input_im_data) = TestInput::builder(client.clone(), signal_shape.clone())
        .dtype(dtype)
        .uniform(7, -1., 1.)
        .generate_with_f32_host_data();

    let spectrum_re = empty_tensor(&client, signal_shape.clone(), dtype);
    let spectrum_im = empty_tensor(&client, signal_shape.clone(), dtype);
    let recovered_re = empty_tensor(&client, signal_shape.clone(), dtype);
    let recovered_im = empty_tensor(&client, signal_shape.clone(), dtype);

    let forward = CfftBindings {
        input_re: input_re.binding(),
        input_im: input_im.binding(),
        output_re: spectrum_re.clone().binding(),
        output_im: spectrum_im.clone().binding(),
    };
    let inverse = CfftBindings {
        input_re: spectrum_re.binding(),
        input_im: spectrum_im.binding(),
        output_re: recovered_re.clone().binding(),
        output_im: recovered_im.clone().binding(),
    };

    let outcome = launch_and_capture_outcome(&client, |c| {
        if let Err(e) =
            cfft_launch_any_size::<TestRuntime>(c, forward, dim, dtype, FftMode::Forward)
        {
            return ExecutionOutcome::CompileError(format!("forward launch failed: {e}"));
        }
        cfft_launch_any_size::<TestRuntime>(c, inverse, dim, dtype, FftMode::Inverse).into()
    });

    match outcome {
        ExecutionOutcome::Executed => {
            let actual_re = HostData::from_tensor_handle(&client, recovered_re, HostDataType::F32);
            let actual_im = HostData::from_tensor_handle(&client, recovered_im, HostDataType::F32);
            let expected_re = scaled(input_re_data, n);
            let expected_im = scaled(input_im_data, n);
            combine_re_im(
                assert_equals_approx(&actual_re, &expected_re, 1e-2),
                assert_equals_approx(&actual_im, &expected_im, 1e-2),
            )
            .as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}

#[test]
fn cfft_roundtrip_axis_last() {
    cfft_roundtrip_case([1, 8].to_vec(), 1);
}

#[test]
fn cfft_roundtrip_axis_1_strided() {
    cfft_roundtrip_case([2, 8, 3].to_vec(), 1);
}

#[test]
fn cfft_roundtrip_axis_0_strided() {
    cfft_roundtrip_case([16, 2].to_vec(), 0);
}

// Batched cases: several windows transformed in one launch, every element
// checked. Guards against per-window state (shared memory) bleeding between
// concurrent cubes, which corrupts all but one window.
#[test]
fn cfft_roundtrip_batched() {
    cfft_roundtrip_case([5, 16].to_vec(), 1);
}

#[test]
fn cfft_roundtrip_batched_larger() {
    cfft_roundtrip_case([4, 64].to_vec(), 1);
}

#[test]
fn cfft_wrapper_roundtrip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let signal_shape = [2, 8].to_vec();
    let dim = 1;
    let n = signal_shape[dim] as f32;

    let (input_re, input_re_data) = TestInput::builder(client.clone(), signal_shape.clone())
        .dtype(dtype)
        .uniform(42, -1., 1.)
        .generate_with_f32_host_data();
    let (input_im, input_im_data) = TestInput::builder(client.clone(), signal_shape.clone())
        .dtype(dtype)
        .uniform(7, -1., 1.)
        .generate_with_f32_host_data();

    let (spectrum_re, spectrum_im) =
        cfft::<TestRuntime>(input_re, input_im, dim, dtype, FftMode::Forward);
    let (recovered_re, recovered_im) =
        cfft::<TestRuntime>(spectrum_re, spectrum_im, dim, dtype, FftMode::Inverse);

    let actual_re = HostData::from_tensor_handle(&client, recovered_re, HostDataType::F32);
    let actual_im = HostData::from_tensor_handle(&client, recovered_im, HostDataType::F32);
    combine_re_im(
        assert_equals_approx(&actual_re, &scaled(input_re_data, n), 1e-2),
        assert_equals_approx(&actual_im, &scaled(input_im_data, n), 1e-2),
    )
    .as_test_outcome()
    .enforce();
}
