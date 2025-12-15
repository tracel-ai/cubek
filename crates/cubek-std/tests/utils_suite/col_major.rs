use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle};
use cubek_std::test_utils::{HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx};

// It's as if the input's buffer gets into_contiguoused while its strides remain non-contiguous,
// somewhere when launching the kernel.

#[cube(launch_unchecked)]
fn copy_matrix_strided<T: Numeric>(
    input: &Tensor<T>,
    out: &mut Tensor<T>,
    #[define(T)] _input_dtype: StorageType,
) {
    let stride_row_in = input.stride(0);
    let stride_col_in = input.stride(1);
    let stride_row_out = out.stride(0);
    let stride_col_out = out.stride(1);

    for i in 0..3 {
        for j in 0..3 {
            out[i * stride_row_out + j * stride_col_out] =
                input[i * stride_row_in + j * stride_col_in]
        }
    }
}

#[test]
fn test_col_major() {
    let client = TestRuntime::client(&Default::default());

    let (input, input_data) = TestInput::arange(
        client.clone(),
        vec![3, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();
    let input_ref = input.as_ref();

    let out = TestInput::zeros(client.clone(), vec![3, 3], f32::as_type_native_unchecked())
        .generate_without_host_data()
        .unwrap();
    let out_ref = out.as_ref();

    let cube_dim = CubeDim::new_single();
    let cube_count = CubeCount::new_single();

    let result = unsafe {
        copy_matrix_strided::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            input_ref.as_tensor_arg(1),
            out_ref.as_tensor_arg(1),
            f32::as_type_native_unchecked(),
        )
    };

    let output_data = HostData::from_tensor_handle(
        &client,
        &TensorHandle::from_ref(&out_ref, f32::as_type_native_unchecked()),
        HostDataType::F32,
    );

    println!("{:?}", input_data);
    println!("{:?}", output_data);

    match assert_equals_approx(&output_data, &input_data, 0.01) {
        Ok(_) => {}
        Err(_) => panic!(),
    }
}
