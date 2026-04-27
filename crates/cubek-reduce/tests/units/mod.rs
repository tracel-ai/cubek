use cubecl::features::Plane;
use cubecl::frontend::CubePrimitive;
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime, cube, ir::StorageType, prelude::*,
    std::tensor::TensorHandle, zspace::Shape,
};
use cubek_reduce::components::instructions::{plane_topk_insert, plane_topk_merge};
use cubek_test_utils::{DataKind, InputDataType, StrideSpec, TestInput};

use crate::it::reference::contiguous_strides;
use cubecl::frontend::CompilationArg;

#[test]
fn test_plane_reduce_inplace() {
    let client = TestRuntime::client(&Default::default());
    if !client.properties().features.plane.contains(Plane::Ops) {
        return;
    }

    // plane_size of 16 with vector_size of 4
    let num_threads = 2;
    let k = 2;
    let vector_size = 4;
    let total_vectors = num_threads * k * vector_size;

    let shape = Shape::new([total_vectors]);
    let stride = contiguous_strides(&shape);

    let dtype = f32::as_type_native_unchecked().storage_type();
    let input_dtype = InputDataType::Standard(dtype);

    #[rustfmt::skip]
    let data = vec![
        // Thread 0
        99.0, 99.1, 99.2, 99.3, 
        10.0, 10.1, 10.2, 10.3, 
        // Thread 1
        88.0, 88.1, 102.2, 88.3, 
        55.0, 55.1, 101.2, 55.3,
    ];

    let (input_handle, _input_host) = TestInput::new(
        client.clone(),
        shape.clone(),
        input_dtype,
        StrideSpec::Custom(stride.iter().copied().collect()),
        DataKind::Custom { data: data.clone() },
    )
    .generate_with_f32_host_data();

    let storage_type = f32::as_type_native_unchecked().storage_type();

    let output_handle = build_output_tensor(&client, storage_type, &shape);

    launch_plane_reduce_inplace::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, num_threads),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
        k,
        storage_type,
        vector_size,
    );

    let bytes = client.read_one(output_handle.handle).unwrap();
    let actual = f32::from_bytes(&bytes);
    assert_plane_topk_custom_values(&data, actual, num_threads, k, vector_size);
}

fn build_output_tensor(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    output_dtype: StorageType,
    output_shape: &Shape,
) -> TensorHandle<TestRuntime> {
    let strides = contiguous_strides(output_shape);
    TestInput::new(
        client.clone(),
        output_shape.clone(),
        output_dtype,
        StrideSpec::Custom(strides.iter().copied().collect()),
        DataKind::Zeros,
    )
    .generate()
}

#[cube(launch)]
fn launch_plane_reduce_inplace<N: Numeric, S: Size>(
    input: &Tensor<Vector<N, S>>,
    output: &mut Tensor<Vector<N, S>>,
    #[comptime] k: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    let mut elements = Array::new(k);
    let offset = UNIT_POS_X as usize * k;

    // Load backwards so the accumulator is already sorted descending locally
    #[unroll]
    for i in 0..k {
        elements[i] = input[offset + i];
    }

    plane_topk_merge::<N, S>(k, &mut elements);

    #[unroll]
    for i in 0..k {
        output[offset + i] = elements[i];
    }
}

fn assert_plane_topk_custom_values(
    input_host: &[f32],
    actual_gpu: &[f32],
    num_threads: usize,
    k: usize,
    vector_size: usize,
) {
    let mut expected_topk = vec![0.0; k * vector_size];

    // Sort each lane independently
    for lane in 0..vector_size {
        let mut lane_values = Vec::new();
        for i in 0..(num_threads * k) {
            lane_values.push(input_host[i * vector_size + lane]);
        }

        // Sort descending for this specific lane
        lane_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for i in 0..k {
            expected_topk[i * vector_size + lane] = lane_values[i];
        }
    }

    for unit in 0..num_threads {
        let start = unit * k * vector_size;
        let end = start + (k * vector_size);
        assert_eq!(&actual_gpu[start..end], expected_topk.as_slice());
    }
}

#[test]
fn test_plane_topk_insert() {
    let client = TestRuntime::client(&Default::default());
    if !client.properties().features.plane.contains(Plane::Ops) {
        return;
    }

    let num_threads = 2;
    let k = 2;
    let vector_size = 4;

    #[rustfmt::skip]
    let acc_data = vec![
        // Thread 0: [100, 10]
        100.0, 100.0, 100.0, 100.0,
        10.0,  10.0,  10.0,  10.0,
        // Thread 1: [50, 5]
        50.0,  50.0,  15.0,  50.0,
        5.0,   5.0,   17.0,   5.0,
    ];

    #[rustfmt::skip]
    let item_data = vec![
        80.0,  80.0,  80.0,  80.0,  // Thread 0 provides 80
        120.0, 120.0, 15.0, 120.0, // Thread 1 provides 120
    ];

    let acc_shape = Shape::new([num_threads * k * vector_size]);
    let acc_stride = contiguous_strides(&acc_shape);
    let item_shape = Shape::new([num_threads * vector_size]);
    let item_stride = contiguous_strides(&item_shape);

    let dtype = f32::as_type_native_unchecked().storage_type();
    let input_dtype = InputDataType::Standard(dtype);

    let (acc_handle, _acc_host) = TestInput::new(
        client.clone(),
        acc_shape.clone(),
        input_dtype.clone(),
        StrideSpec::Custom(acc_stride.iter().copied().collect()),
        DataKind::Custom {
            data: acc_data.clone(),
        },
    )
    .generate_with_f32_host_data();

    let (item_handle, _item_host) = TestInput::new(
        client.clone(),
        item_shape.clone(),
        input_dtype,
        StrideSpec::Custom(item_stride.iter().copied().collect()),
        DataKind::Custom {
            data: item_data.clone(),
        },
    )
    .generate_with_f32_host_data();

    let storage_type = f32::as_type_native_unchecked().storage_type();

    launch_plane_topk_insert::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, num_threads),
        acc_handle.clone().binding().into_tensor_arg(),
        item_handle.binding().into_tensor_arg(),
        k,
        storage_type,
        vector_size,
    );

    let bytes = client.read_one(acc_handle.handle).unwrap();
    let actual = f32::from_bytes(&bytes);

    assert_lane_topk_insert(&acc_data, &item_data, actual, num_threads, k, vector_size);
}

#[cube(launch)]
fn launch_plane_topk_insert<N: Numeric, S: Size>(
    accumulator: &mut Tensor<Vector<N, S>>,
    new_item: &Tensor<Vector<N, S>>,
    #[comptime] k: usize,
    #[define(N)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    let mut elements = Array::new(k);
    let offset = UNIT_POS_X as usize * k;

    #[unroll]
    for i in 0..k {
        elements[i] = accumulator[offset + i];
    }

    let item = new_item[UNIT_POS_X as usize];

    plane_topk_insert::<N, S>(&mut elements, item, k);

    #[unroll]
    for i in 0..k {
        accumulator[offset + i] = elements[i];
    }
}

fn assert_lane_topk_insert(
    initial_acc: &[f32],
    new_items: &[f32],
    actual_gpu: &[f32],
    num_threads: usize,
    k: usize,
    vector_size: usize,
) {
    let mut plane_items_per_lane: Vec<Vec<f32>> = vec![Vec::new(); vector_size];
    for unit in 0..num_threads {
        for s in 0..vector_size {
            plane_items_per_lane[s].push(new_items[unit * vector_size + s]);
        }
    }

    for unit in 0..num_threads {
        for s in 0..vector_size {
            let mut candidates = Vec::new();

            for i in 0..k {
                candidates.push(initial_acc[(unit * k + i) * vector_size + s]);
            }

            candidates.extend_from_slice(&plane_items_per_lane[s]);

            candidates.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let expected_lane_topk = &candidates[..k];

            for i in 0..k {
                let actual_val = actual_gpu[(unit * k + i) * vector_size + s];
                assert_eq!(
                    actual_val, expected_lane_topk[i],
                    "Mismatch at Thread {}, Lane {}, Rank {}. Expected {}, got {}",
                    unit, s, i, expected_lane_topk[i], actual_val
                );
            }
        }
    }
}
