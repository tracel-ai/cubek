use super::make_problem;
use crate::pool::{PoolForward, Pooled, Seeded};
use cubecl::zspace::Shape;
use cubek_pool::{definition::AvgPoolOptions, eval::cpu_reference::geometry::PoolGeometry};
use cubek_test_utils::ScopeUnderTest;

const AVG_POOL2D_TOLERANCE: f32 = 1e-5;

#[test]
fn test_avg_pool2d_basic() {
    let problem = make_problem(
        Shape::from([2, 4, 4, 2]),
        false,
        AvgPoolOptions::new([3, 3], [1, 1], [1, 1], false, false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 5678,
            inputs: -10.0..10.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_include_pad() {
    // count_include_pad = true: divisor is always kernel_size * kernel_size
    let problem = make_problem(
        Shape::from([1, 5, 5, 3]),
        false,
        AvgPoolOptions::new([3, 3], [2, 2], [1, 1], false, true),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 9999,
            inputs: 0.0..1.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_exclude_pad() {
    // count_include_pad = false: divisor only counts non-padded elements
    let problem = make_problem(
        Shape::from([1, 5, 5, 3]),
        false,
        AvgPoolOptions::new([3, 3], [2, 2], [1, 1], false, false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 8888,
            inputs: 0.0..1.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_strided_no_pad() {
    let problem = make_problem(
        Shape::from([2, 6, 6, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], false, false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 1234,
            inputs: -1.0..1.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_non_square_asymmetric() {
    let problem = make_problem(
        Shape::from([2, 5, 7, 3]),
        false,
        AvgPoolOptions::new([2, 3], [1, 2], [1, 0], false, false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 3456,
            inputs: -1.0..1.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_ceil_mode() {
    let problem = make_problem(
        Shape::from([2, 5, 5, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], true, false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 4567,
            inputs: -1.0..1.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_large_kernel() {
    let problem = make_problem(
        Shape::from([1, 10, 10, 1]),
        false,
        AvgPoolOptions::new([7, 7], [1, 1], [0, 0], false, false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 7777,
            inputs: -5.0..5.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_dilation_include_pad_ceil_mode() {
    run_dilated_ceil_case(true);
}

#[test]
fn test_avg_pool2d_dilation_exclude_pad_ceil_mode() {
    run_dilated_ceil_case(false);
}

fn run_dilated_ceil_case(count_include_pad: bool) {
    // Unequal dilation, overlapping windows, explicit padding and a trailing tap beyond it.
    let problem = make_problem(
        Shape::from([2, 5, 6, 4]),
        false,
        AvgPoolOptions::new([2, 3], [2, 2], [1, 2], true, count_include_pad).with_dilation([2, 3]),
    );
    assert_eq!(
        problem.output_shape(&problem.input_shape),
        Shape::from([2, 3, 3, 4])
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 2468,
            inputs: -10.0..10.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_dilation_no_valid_taps() {
    // Both taps on each axis miss the sole input position. Excluding padding yields zero.
    let problem = make_problem(
        Shape::from([2, 1, 1, 3]),
        false,
        AvgPoolOptions::new([2, 2], [1, 1], [1, 1], false, false).with_dilation([2, 2]),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 2469,
            inputs: 1.0..10.0,
        },
        tolerance: AVG_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_avg_pool2d_rejects_zero_dilation() {
    use crate::pool::build_output_tensor;
    use cubecl::prelude::*;
    use cubek_pool::{definition::PoolError, pool2d, pool2d_backward};

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let tensor = build_output_tensor(&client, vec![1, 1, 1, 1], dtype);
    let options = AvgPoolOptions::new([1, 1], [1, 1], [0, 0], false, false).with_dilation([0, 1]);
    assert!(matches!(
        pool2d(
            &client,
            tensor.clone().binding(),
            tensor.clone().binding(),
            options.clone().into(),
            dtype,
        ),
        Err(PoolError::InvalidDilation { .. })
    ));
    assert!(matches!(
        pool2d_backward(
            &client,
            tensor.clone().binding(),
            tensor.clone().binding(),
            tensor.binding(),
            options.into(),
            dtype,
        ),
        Err(PoolError::InvalidDilation { .. })
    ));
}
