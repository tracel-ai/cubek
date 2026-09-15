use super::{
    build_output_tensor, make_problem, output_host_f32, run_pool_backward_test, validate_test,
};
use cubecl::zspace::Shape;
use cubek_pool::{
    definition::AvgPoolOptions, eval::cpu_reference::cpu_reference_pool_backward, pool2d_backward,
};
use cubek_test_utils::TestInput;

const AVG_POOL2D_BACKWARD_TOLERANCE: f32 = 0.000001;

#[test]
fn test_avg_pool2d_backward() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [4, 4],
        Shape::from([2, 4, 4, 2]),
        false,
        AvgPoolOptions::new([3, 3], [1, 1], [1, 1], false, false),
    );
    run_pool_backward_test(
        client,
        5678,
        -10.0,
        10.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_strided_no_pad() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [6, 6],
        Shape::from([2, 3, 3, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], false, false),
    );
    run_pool_backward_test(
        client,
        1234,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_exclude_pad() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [5, 5],
        Shape::from([1, 3, 3, 1]),
        false,
        AvgPoolOptions::new([3, 3], [2, 2], [1, 1], false, false),
    );
    run_pool_backward_test(
        client,
        9999,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_non_square_asymmetric() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [5, 7],
        Shape::from([2, 6, 3, 3]),
        false,
        AvgPoolOptions::new([2, 3], [1, 2], [1, 0], false, true),
    );
    run_pool_backward_test(
        client,
        3456,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_ceil_mode() {
    let client = cubecl::test_device().client();
    for count_include_pad in [true, false] {
        for (input_size, out_size, kernel, padding, upstream, expected) in [
            (
                [1, 5],
                [1, 3],
                [1, 2],
                [0, 0],
                vec![2.0, 4.0, 8.0],
                vec![1.0, 1.0, 2.0, 2.0, 8.0],
            ),
            // Padded counts: [9, 6, 6, 4]; valid counts: [4, 2, 2, 1].
            (
                [3, 3],
                [2, 2],
                [3, 3],
                [1, 1],
                vec![36.0, 24.0, 24.0, 16.0],
                if count_include_pad {
                    vec![4.0; 9]
                } else {
                    vec![9.0, 9.0, 12.0, 9.0, 9.0, 12.0, 12.0, 12.0, 16.0]
                },
            ),
        ] {
            let shape = [1, input_size[0], input_size[1], 1];
            let out_shape = Shape::from([1, out_size[0], out_size[1], 1]);
            let problem = make_problem(
                input_size,
                out_shape.clone(),
                false,
                AvgPoolOptions::new(kernel, kernel, padding, true, count_include_pad),
            );
            let (input, input_data) = TestInput::builder(client.clone(), shape)
                .zeros()
                .generate_with_f32_host_data();
            let (grad, grad_data) = TestInput::builder(client.clone(), out_shape)
                .custom(upstream)
                .generate_with_f32_host_data();
            let expected = TestInput::builder(client.clone(), shape)
                .custom(expected)
                .f32_host_data();
            let output = build_output_tensor(&client, shape.to_vec(), input.dtype);
            pool2d_backward(
                &client,
                input.clone().binding(),
                grad.binding(),
                output.clone().binding(),
                problem.mode.clone(),
                input.dtype,
            )
            .unwrap();
            validate_test(
                Ok(()),
                output_host_f32(&client, output),
                expected.clone(),
                AVG_POOL2D_BACKWARD_TOLERANCE,
            );
            validate_test(
                Ok(()),
                cpu_reference_pool_backward(&grad_data, &input_data, problem),
                expected,
                AVG_POOL2D_BACKWARD_TOLERANCE,
            );
        }
    }
}
