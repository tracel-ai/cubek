use cubecl::{
    Runtime, TestRuntime,
    client::ComputeClient,
    frontend::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use cubek_reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy},
};
use cubek_reduce::{reduce, routines::unit::UnitStrategy};
use cubek_test_utils::{
    DataKind, ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput,
    TestOutcome, ValidationResult, assert_equals_approx,
};

#[test]
fn simple_reduce_sum() {
    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
            independent: false,
        })),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let shape = [32, 512].to_vec();
    let dim = 1;
    test_launch(client, shape, dim, strategy);
}

#[test]
fn simple_reduce_sum_strategy_unit() {
    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy {})),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let shape = [32, 128].to_vec();
    let dim = 1;
    test_launch(client, shape, dim, strategy);
}

#[test]
fn simple_reduce_sum_strategy_cube_use_planes() {
    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: true,
        })),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let shape = [32, 128].to_vec();
    let dim = 1;
    test_launch(client, shape, dim, strategy);
}

#[test]
fn simple_reduce_sum_strategy_cube_not_use_planes() {
    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: false,
        })),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let shape = [4, 128].to_vec();
    let dim = 1;
    test_launch(client, shape, dim, strategy);
}

#[test]
fn simple_reduce_sum_strategy_cube_use_planes_perpendicular() {
    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: false,
        })),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let shape = [4, 128].to_vec();
    let dim = 1;
    test_launch(client, shape, dim, strategy);
}

fn test_launch(
    client: ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dim: usize,
    strategy: ReduceStrategy,
) {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let mut reduce_shape = shape.clone();
    reduce_shape[dim] = 1;

    let (matrix_handle, matrix_data) = TestInput::new(
        client.clone(),
        shape.clone(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Arange { scale: Some(1.) },
    )
    .generate_with_f32_host_data();

    let reduce_matrix_handle = TestInput::new(
        client.clone(),
        reduce_shape.to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    match reduce::<TestRuntime>(
        &client,
        matrix_handle.binding(),
        reduce_matrix_handle.clone().binding(),
        dim,
        strategy,
        ReduceOperationConfig::Sum,
        ReduceDtypes {
            input: f32::as_type_native_unchecked().storage_type(),
            output: f32::as_type_native_unchecked().storage_type(),
            accumulation: f32::as_type_native_unchecked().storage_type(),
        },
    )
    .into()
    {
        ExecutionOutcome::Executed => {
            assert_reduce_sum_result(&client, matrix_data, reduce_matrix_handle, dim)
                .as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}

use ValidationResult::*;

pub fn assert_reduce_sum_result(
    client: &ComputeClient<TestRuntime>,
    matrix: HostData,
    reduce_matrix: TensorHandle<TestRuntime>,
    dim: usize,
) -> ValidationResult {
    // big epsilon because with wgpu, compute is less precise
    let epsilon = 0.2;
    let expected_reduce = reduce_sum_ref(&matrix, dim);

    let actual_reduce = HostData::from_tensor_handle(client, reduce_matrix, HostDataType::F32);

    let result_reduce = assert_equals_approx(&actual_reduce, &expected_reduce, epsilon);

    match result_reduce {
        Fail(e) => Fail(e.clone()),
        Skipped(r1) => Skipped(r1.to_string()),
        Pass => Pass,
        _ => panic!("unreachable"),
    }
}

// Reference code
pub fn reduce_sum_ref(matrix: &HostData, dim: usize) -> HostData {
    let in_shape = matrix.shape.as_slice();
    let reduce_len = in_shape[dim];

    let mut out_shape_vec = in_shape.to_vec();
    out_shape_vec[dim] = 1;
    let out_shape = Shape::from(out_shape_vec);

    let num_reductions = matrix.shape.num_elements() / reduce_len;
    let out_strides = StrideSpec::RowMajor.compute_strides(&out_shape);

    let mut flattened = vec![0.0; out_shape.num_elements()];

    for l in 0..num_reductions {
        let mut coords = get_coords(l, in_shape, dim);

        let mut sum = 0.0;
        for i in 0..reduce_len {
            coords[dim] = i;
            sum += matrix.get_f32(&coords);
        }

        coords[dim] = 0;
        let flat_idx = compute_index(&out_strides, coords.as_slice());
        flattened[flat_idx] = sum;
    }

    HostData {
        data: HostDataVec::F32(flattened),
        shape: out_shape,
        strides: out_strides,
    }
}

fn get_coords(lane_idx: usize, shape: &[usize], dim: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut temp = lane_idx;
    for i in (0..shape.len()).rev() {
        if i == dim {
            continue;
        }
        coords[i] = temp % shape[i];
        temp /= shape[i];
    }
    coords
}

pub fn compute_index(strides: &Strides, coords: &[usize]) -> usize {
    assert_eq!(
        coords.len(),
        strides.rank(),
        "Coordinate rank must match stride rank",
    );

    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}
