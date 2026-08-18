//! Unit tests for leaf microkernels in `instruction/`.

use cubecl::{Runtime, TestRuntime, client::ComputeClient, prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::instruction::{logsumexp, max, min, sum};

#[cube(launch)]
fn test_hsum_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    let size!(W4) = 4;
    let size!(W2) = 2;
    let mut v4 = Vector::<f32, W4>::zeroed();
    v4.insert(0usize, input[0]);
    v4.insert(1usize, input[1]);
    v4.insert(2usize, input[2]);
    v4.insert(3usize, input[3]);

    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, input[0]);
    v2.insert(1usize, input[1]);

    output[0] = sum::vector(v4, 4usize);
    output[1] = sum::vector(v4, 2usize);
    output[2] = sum::vector(v2, 2usize);

    let mut arr = Array::<f32>::new(4usize);
    arr[0] = 1.0f32;
    arr[1] = 2.0f32;
    arr[2] = 3.0f32;
    arr[3] = 4.0f32;
    output[3] = sum::array(&arr, 4usize);
    output[4] = sum::array(&arr, 2usize);
    output[5] = sum::array_from(&arr, 4usize, 5.0f32);
}

#[cube(launch)]
fn test_extrema_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    let size!(W4) = 4;
    let size!(W2) = 2;
    let mut v4 = Vector::<f32, W4>::zeroed();
    v4.insert(0usize, input[0]);
    v4.insert(1usize, input[1]);
    v4.insert(2usize, input[2]);
    v4.insert(3usize, input[3]);

    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, input[0]);
    v2.insert(1usize, input[1]);

    output[0] = max::vector(v4, 4usize);
    output[1] = min::vector(v4, 4usize);
    output[2] = max::vector(v2, 2usize);
    output[3] = min::vector(v2, 2usize);

    let mut arr = Array::<f32>::new(4usize);
    arr[0] = 3.0f32;
    arr[1] = 1.0f32;
    arr[2] = 7.0f32;
    arr[3] = 2.0f32;
    output[4] = max::array(&arr, 4usize);
    output[5] = min::array(&arr, 4usize);
    output[6] = max::array_from(&arr, 4usize, 10.0f32);
    output[7] = min::array_from(&arr, 4usize, -10.0f32);
}

#[cube(launch)]
fn test_logsumexp_step_kernel(scores: &Tensor<f32>, output: &mut Tensor<f32>) {
    let m_init = f32::min_value();
    let l_init = 0.0f32;

    let (m0, l0, corr0, w0) = logsumexp::step::<f32>(m_init, l_init, scores[0]);
    output[0] = m0;
    output[1] = l0;
    output[2] = corr0;
    output[3] = w0;

    let (m1, l1, corr1, w1) = logsumexp::step::<f32>(m0, l0, scores[1]);
    output[4] = m1;
    output[5] = l1;
    output[6] = corr1;
    output[7] = w1;
}

#[cube(launch)]
fn test_plane_and_group_kernel(output: &mut Tensor<f32>) {
    let lane_id = UNIT_POS_X;
    let val = (lane_id + 1u32) as f32; // Lane 0: 1.0, Lane 1: 2.0, Lane 2: 3.0, Lane 3: 4.0

    // Non-trivial 4-lane plane operations
    let p_sum = sum::plane::<f32>(val, 4usize);
    let p_max = max::plane::<f32>(val, 4usize);
    let p_min = min::plane::<f32>(val, 4usize);

    // Non-trivial 4-lane butterfly group fold (mask 0b11 = folds all 4 lanes)
    let size!(W2) = 2;
    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, val);
    v2.insert(1usize, val * 2.0f32);
    let folded_full = sum::group::<f32, W2>(v2, 0b11usize);

    // 2-lane sub-group butterfly fold (mask 0b01 = folds (0,1) and (2,3) separately)
    let folded_pair = sum::group::<f32, W2>(v2, 0b01usize);

    // The same butterfly under max and min
    let max_full = max::group::<f32, W2>(v2, 0b11usize);
    let min_full = min::group::<f32, W2>(v2, 0b11usize);
    let min_pair = min::group::<f32, W2>(v2, 0b01usize);

    // 1-lane fallback paths (lanes = 1, mask = 0)
    let s_fallback = sum::plane::<f32>(val, 1usize);
    let m_fallback = max::plane::<f32>(val, 1usize);
    let n_fallback = min::plane::<f32>(val, 1usize);
    let g_fallback = sum::group::<f32, W2>(v2, 0usize);

    // Store per-lane results at lane_id * 15
    let base = (lane_id * 15u32) as usize;
    output[base] = p_sum;
    output[base + 1] = p_max;
    output[base + 2] = p_min;
    output[base + 3] = folded_full.extract(0usize);
    output[base + 4] = folded_full.extract(1usize);
    output[base + 5] = folded_pair.extract(0usize);
    output[base + 6] = folded_pair.extract(1usize);
    output[base + 7] = s_fallback;
    output[base + 8] = m_fallback;
    output[base + 9] = n_fallback;
    output[base + 10] = g_fallback.extract(0usize);
    output[base + 11] = max_full.extract(0usize);
    output[base + 12] = max_full.extract(1usize);
    output[base + 13] = min_full.extract(0usize);
    output[base + 14] = min_pair.extract(0usize);
}

/// CPU planes contain one unit, so only exercise the explicit one-lane fallbacks there.
#[cube(launch)]
fn test_plane_and_group_fallback_kernel(output: &mut Tensor<f32>) {
    let lane_id = UNIT_POS_X;
    let val = (lane_id + 1u32) as f32;
    let size!(W2) = 2;
    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, val);
    v2.insert(1usize, val * 2.0f32);

    let base = (lane_id * 15u32) as usize;
    output[base + 7] = sum::plane::<f32>(val, 1usize);
    output[base + 8] = max::plane::<f32>(val, 1usize);
    output[base + 9] = min::plane::<f32>(val, 1usize);
    output[base + 10] = sum::group::<f32, W2>(v2, 0usize).extract(0usize);
}

#[test]
fn test_hsum_and_array_sum() {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let (input_handle, _data) = TestInput::builder(client.clone(), Shape::new([4]))
        .dtype(f32::elem_type_native())
        .custom(vec![1.0, 2.0, 3.0, 4.0])
        .generate_with_f32_host_data();
    let output_handle = TestInput::builder(client.clone(), Shape::new([6]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    test_hsum_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
    assert_eq!(output.get_f32(&[0]), 10.0); // 1 + 2 + 3 + 4
    assert_eq!(output.get_f32(&[1]), 3.0); // 1 + 2
    assert_eq!(output.get_f32(&[2]), 3.0); // 1 + 2
    assert_eq!(output.get_f32(&[3]), 10.0); // 1 + 2 + 3 + 4 (identity seeded)
    assert_eq!(output.get_f32(&[4]), 3.0); // 1 + 2 (identity seeded)
    assert_eq!(output.get_f32(&[5]), 15.0); // 5 + 1 + 2 + 3 + 4 (seeded)
}

#[test]
fn test_extrema_max_min() {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let (input_handle, _data) = TestInput::builder(client.clone(), Shape::new([4]))
        .dtype(f32::elem_type_native())
        .custom(vec![3.0, 1.0, 7.0, 2.0])
        .generate_with_f32_host_data();
    let output_handle = TestInput::builder(client.clone(), Shape::new([8]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    test_extrema_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
    assert_eq!(output.get_f32(&[0]), 7.0); // max(3, 1, 7, 2)
    assert_eq!(output.get_f32(&[1]), 1.0); // min(3, 1, 7, 2)
    assert_eq!(output.get_f32(&[2]), 3.0); // max(3, 1)
    assert_eq!(output.get_f32(&[3]), 1.0); // min(3, 1)
    assert_eq!(output.get_f32(&[4]), 7.0); // max array (identity seeded with min_value)
    assert_eq!(output.get_f32(&[5]), 1.0); // min array (identity seeded with max_value)
    assert_eq!(output.get_f32(&[6]), 10.0); // max array starting from 10.0
    assert_eq!(output.get_f32(&[7]), -10.0); // min array starting from -10.0
}

#[test]
fn test_logsumexp_step() {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let (input_handle, _data) = TestInput::builder(client.clone(), Shape::new([2]))
        .dtype(f32::elem_type_native())
        .custom(vec![2.0, 5.0])
        .generate_with_f32_host_data();
    let output_handle = TestInput::builder(client.clone(), Shape::new([8]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    test_logsumexp_step_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

    // Step 0: score 2.0 -> m = 2.0, l = exp(2-2) = 1.0, w = 1.0
    assert_eq!(output.get_f32(&[0]), 2.0);
    assert_eq!(output.get_f32(&[1]), 1.0);
    assert_eq!(output.get_f32(&[3]), 1.0);

    // Step 1: score 5.0 -> m = 5.0, corr = exp(2-5) = exp(-3), w = exp(5-5) = 1.0, l = 1.0*exp(-3) + 1.0
    assert_eq!(output.get_f32(&[4]), 5.0);
    let expected_corr = (-3.0f32).exp();
    let expected_l = 1.0 * expected_corr + 1.0;
    assert!((output.get_f32(&[5]) - expected_l).abs() < 1e-6);
    assert!((output.get_f32(&[6]) - expected_corr).abs() < 1e-6);
    assert_eq!(output.get_f32(&[7]), 1.0);
}

#[test]
fn test_plane_and_group_primitives() {
    let client: ComputeClient<TestRuntime> = <TestRuntime as Runtime>::client(&Default::default());
    let output_handle = TestInput::builder(client.clone(), Shape::new([60]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let is_cpu = client.properties().hardware.num_cpu_cores.is_some();
    if is_cpu {
        test_plane_and_group_fallback_kernel::launch::<TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            output_handle.clone().binding().into_tensor_arg(),
        );
    } else {
        test_plane_and_group_kernel::launch::<TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            output_handle.clone().binding().into_tensor_arg(),
        );
    }

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

    for lane in 0..4 {
        let base = lane * 15;
        let val = (lane + 1) as f32;

        if !is_cpu {
            // Plane cooperative intrinsics (all lanes see the whole-plane reduction)
            assert_eq!(output.get_f32(&[base]), 10.0, "plane_sum on lane {lane}");
            assert_eq!(output.get_f32(&[base + 1]), 4.0, "plane_max on lane {lane}");
            assert_eq!(output.get_f32(&[base + 2]), 1.0, "plane_min on lane {lane}");

            // 4-lane butterfly group fold (mask 0b11: all lanes hold the 4-lane vector total)
            assert_eq!(
                output.get_f32(&[base + 3]),
                10.0,
                "fold_group 0b11 [0] on lane {lane}"
            );
            assert_eq!(
                output.get_f32(&[base + 4]),
                20.0,
                "fold_group 0b11 [1] on lane {lane}"
            );

            // 2-lane pairwise butterfly fold (mask 0b01: lanes (0,1) and (2,3) fold separately)
            if lane < 2 {
                assert_eq!(
                    output.get_f32(&[base + 5]),
                    3.0,
                    "fold_group 0b01 [0] on lane {lane}"
                );
                assert_eq!(
                    output.get_f32(&[base + 6]),
                    6.0,
                    "fold_group 0b01 [1] on lane {lane}"
                );
            } else {
                assert_eq!(
                    output.get_f32(&[base + 5]),
                    7.0,
                    "fold_group 0b01 [0] on lane {lane}"
                );
                assert_eq!(
                    output.get_f32(&[base + 6]),
                    14.0,
                    "fold_group 0b01 [1] on lane {lane}"
                );
            }
        }

        // 1-lane fallback paths (lanes = 1, mask = 0)
        assert_eq!(
            output.get_f32(&[base + 7]),
            val,
            "sum 1-lane fallback on lane {lane}"
        );
        assert_eq!(
            output.get_f32(&[base + 8]),
            val,
            "max 1-lane fallback on lane {lane}"
        );
        assert_eq!(
            output.get_f32(&[base + 9]),
            val,
            "min 1-lane fallback on lane {lane}"
        );
        assert_eq!(
            output.get_f32(&[base + 10]),
            val,
            "group mask 0 fallback on lane {lane}"
        );

        if !is_cpu {
            // Max/min butterfly over the same 4 lanes: vectors are [1,2] [2,4] [3,6] [4,8]
            assert_eq!(
                output.get_f32(&[base + 11]),
                4.0,
                "max_group 0b11 [0] on lane {lane}"
            );
            assert_eq!(
                output.get_f32(&[base + 12]),
                8.0,
                "max_group 0b11 [1] on lane {lane}"
            );
            assert_eq!(
                output.get_f32(&[base + 13]),
                1.0,
                "min_group 0b11 [0] on lane {lane}"
            );
            assert_eq!(
                output.get_f32(&[base + 14]),
                if lane < 2 { 1.0 } else { 3.0 },
                "min_group 0b01 [0] on lane {lane}"
            );
        }
    }
}
