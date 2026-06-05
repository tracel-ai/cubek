use cubecl::prelude::*;
use cubek_resample::components::{NdLayoutLaunch, resample_kernel};
use cubek_resample::definition::{GlobalOp, IdentityCombine, LinearReduction, NearestMapper};

#[derive(Clone, Copy)]
pub struct TestOp;

impl GlobalOp for TestOp {
    type F = NearestMapper; // we use nearest mapper for everything just to test it
    type H = NearestMapper;
    type K = NearestMapper;
    type Combine = IdentityCombine;
    type Reduce = LinearReduction;
}

fn test_resample_1d<R: Runtime>() {
    let client = R::client(&Default::default());

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
        f32::as_bytes(&input_data).to_vec(),
    ));
    let input =
        unsafe { TensorArg::<R>::from_raw_parts(input_handle.clone(), [1].into(), [4].into()) };

    let output_handle = client.empty(8 * core::mem::size_of::<f32>());
    let output =
        unsafe { TensorArg::<R>::from_raw_parts(output_handle.clone(), [1].into(), [8].into()) };

    let mut in_divmods = SequenceArg::new();
    in_divmods.push(4usize);
    let mut in_strides = SequenceArg::new();
    in_strides.push(1usize);
    let in_layout = NdLayoutLaunch::new(in_divmods, in_strides);

    let mut out_divmods = SequenceArg::new();
    out_divmods.push(8usize);
    let mut out_strides = SequenceArg::new();
    out_strides.push(1usize);
    let out_layout = NdLayoutLaunch::new(out_divmods, out_strides);

    let mut scales = SequenceArg::new();
    scales.push(0.5f32); // Scale input coordinates by 0.5 (output size 8 -> input size 4)

    let cube_dim = CubeDim::new_2d(1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        resample_kernel::launch_unchecked::<f32, TestOp, R>(
            &client, cube_count, cube_dim, input, output, out_layout, in_layout, scales,
        );
    }

    let actual_bytes = client.read_one(output_handle.clone()).unwrap();
    let actual_f32 = f32::from_bytes(&actual_bytes);
    assert_eq!(actual_f32, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
}

#[test]
fn resample_1d_test() {
    test_resample_1d::<cubecl::TestRuntime>();
}

fn test_resample_nhwc_2d<R: Runtime>() {
    let client = R::client(&Default::default());

    // Input shape: N=1, H=2, W=2, C=1
    // Flattened: [1.0, 2.0, 3.0, 4.0]
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
        f32::as_bytes(&input_data).to_vec(),
    ));
    let input = unsafe {
        TensorArg::<R>::from_raw_parts(input_handle.clone(), [1, 1].into(), [4, 1].into())
    };

    // Output shape: N=1, H=4, W=4, C=1
    let out_size = 1 * 4 * 4 * 1;
    let output_handle = client.empty(out_size * core::mem::size_of::<f32>());
    let output = unsafe {
        TensorArg::<R>::from_raw_parts(output_handle.clone(), [1, 1].into(), [16, 1].into())
    };

    // Layout config: Innermost is C (1), then W (2), then H (2), then N (1)
    let mut in_divmods = SequenceArg::new();
    in_divmods.push(1usize); // C
    in_divmods.push(2usize); // W
    in_divmods.push(2usize); // H
    in_divmods.push(1usize); // N

    let mut in_strides = SequenceArg::new();
    in_strides.push(1usize); // stride_c
    in_strides.push(1usize); // stride_w
    in_strides.push(2usize); // stride_h
    in_strides.push(4usize); // stride_n
    let in_layout = NdLayoutLaunch::new(in_divmods, in_strides);

    // Output layout config: Innermost is C (1), then W (4), then H (4), then N (1)
    let mut out_divmods = SequenceArg::new();
    out_divmods.push(1usize); // C
    out_divmods.push(4usize); // W
    out_divmods.push(4usize); // H
    out_divmods.push(1usize); // N

    let mut out_strides = SequenceArg::new();
    out_strides.push(1usize); // stride_c
    out_strides.push(1usize); // stride_w
    out_strides.push(4usize); // stride_h
    out_strides.push(16usize); // stride_n
    let out_layout = NdLayoutLaunch::new(out_divmods, out_strides);

    let mut scales = SequenceArg::new();
    scales.push(1.0f32); // scale_c
    scales.push(0.5f32); // scale_w (4 -> 2)
    scales.push(0.5f32); // scale_h (4 -> 2)
    scales.push(1.0f32); // scale_n

    let cube_dim = CubeDim::new_2d(1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        resample_kernel::launch_unchecked::<f32, TestOp, R>(
            &client, cube_count, cube_dim, input, output, out_layout, in_layout, scales,
        );
    }

    let actual_bytes = client.read_one(output_handle.clone()).unwrap();
    let actual_f32 = f32::from_bytes(&actual_bytes);
    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(actual_f32, expected.as_slice());
}

#[test]
fn resample_nhwc_2d_test() {
    test_resample_nhwc_2d::<cubecl::TestRuntime>();
}
