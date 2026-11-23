use super::make_problem;
use crate::pool::{PoolForward, Pooled, Seeded};
use cubecl::zspace::Shape;
use cubek_pool::definition::MaxPoolOptions;
use cubek_test_utils::ScopeUnderTest;

const MAX_POOL2D_TOLERANCE: f32 = 0.0;

#[test]
fn test_max_pool2d() {
    let problem = make_problem(
        Shape::from([2, 4, 4, 2]),
        false,
        MaxPoolOptions::new([3, 3], [1, 1], [1, 1], [1, 1], false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 5678,
            inputs: -10.0..10.0,
        },
        tolerance: MAX_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_max_pool2d_strided_no_pad() {
    let problem = make_problem(
        Shape::from([2, 6, 6, 4]),
        false,
        MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 1234,
            inputs: -1.0..1.0,
        },
        tolerance: MAX_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_max_pool2d_dilated() {
    let problem = make_problem(
        Shape::from([1, 8, 8, 3]),
        false,
        MaxPoolOptions::new([3, 3], [1, 1], [0, 0], [2, 2], false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 2345,
            inputs: -1.0..1.0,
        },
        tolerance: MAX_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_max_pool2d_non_square_asymmetric() {
    let problem = make_problem(
        Shape::from([2, 5, 7, 3]),
        false,
        MaxPoolOptions::new([2, 3], [1, 2], [1, 0], [1, 1], false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 3456,
            inputs: -1.0..1.0,
        },
        tolerance: MAX_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_max_pool2d_ceil_mode() {
    let problem = make_problem(
        Shape::from([2, 5, 5, 4]),
        false,
        MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], true),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 4567,
            inputs: -1.0..1.0,
        },
        tolerance: MAX_POOL2D_TOLERANCE,
    }]);
}

#[test]
fn test_max_pool2d_with_indices() {
    let problem = make_problem(
        Shape::from([1, 3, 3, 2]),
        true,
        MaxPoolOptions::new([2, 2], [1, 1], [0, 0], [1, 1], false),
    );
    PoolForward.check([Pooled {
        problem,
        input: Seeded {
            seed: 6789,
            inputs: -1.0..1.0,
        },
        tolerance: MAX_POOL2D_TOLERANCE,
    }]);
}

/// `values`, laid out as `shape` (NHWC), max pooled with indices under `options`: the output
/// and the argmax, read back in the output's NHWC order.
fn pooled_with_indices(
    shape: [usize; 4],
    values: Vec<f32>,
    options: MaxPoolOptions<2>,
) -> (Vec<f32>, Vec<f32>, cubek_test_utils::HostData, Vec<usize>) {
    use crate::pool::{Entries, INDICES, PoolEntry};
    use cubek_pool::eval::cpu_reference::geometry::PoolGeometry;
    use cubek_test_utils::{HostData, HostDataType, TestInput};

    let client = cubecl::test_device().client();
    let (input, host) = TestInput::builder(client.clone(), shape.to_vec())
        .custom(values)
        .generate_with_f32_host_data();
    let problem = make_problem(Shape::from(shape), true, options.clone());
    let out_shape = problem.output_shape(&problem.input_shape).to_vec();
    let output = Seeded::poisoned(&client, &out_shape, input.dtype);
    let indices = Seeded::poisoned(&client, &out_shape, INDICES);
    Entries::forward(
        &client,
        &input,
        &output,
        Some(&indices),
        problem.mode.clone(),
    )
    .expect("the max pool serves this case");

    let read = |tensor| {
        let host = HostData::from_tensor_handle(&client, tensor, HostDataType::F32);
        let n: usize = out_shape.iter().product();
        (0..n)
            .map(|i| {
                let c = i % out_shape[3];
                let w = i / out_shape[3] % out_shape[2];
                let h = i / (out_shape[3] * out_shape[2]) % out_shape[1];
                let b = i / (out_shape[3] * out_shape[2] * out_shape[1]);
                host.get_f32(&[b, h, w, c])
            })
            .collect::<Vec<_>>()
    };
    (read(output), read(indices), host, out_shape)
}

/// A window of padding alone holds no input, so it is `-inf`, as in PyTorch, not the type's
/// lowest finite value. Two rows and columns of input, padded by one and dilated by three: the one
/// window reads rows and columns -1 and 2.
#[test]
fn test_max_pool2d_with_indices_all_padding_window() {
    let (values, _, _, _) = pooled_with_indices(
        [1, 2, 2, 1],
        vec![1.0, 2.0, 3.0, 4.0],
        MaxPoolOptions::new([2, 2], [1, 1], [1, 1], [3, 3], false),
    );
    assert_eq!(values, [f32::NEG_INFINITY]);
}

/// A window of `-inf` values is `-inf`, at the first value's index.
#[test]
fn test_max_pool2d_with_indices_neg_infinity_window() {
    let (values, indices, _, _) = pooled_with_indices(
        [1, 2, 2, 1],
        vec![f32::NEG_INFINITY; 4],
        MaxPoolOptions::new([2, 2], [1, 1], [0, 0], [1, 1], false),
    );
    assert_eq!(values, [f32::NEG_INFINITY]);
    assert_eq!(indices, [0.0]);
}

/// A NaN in a window is that window's maximum, at the NaN's index, as in PyTorch
/// (`val > max || isnan(val)`): two of the four `2 × 2` windows hold one, first and last in them.
#[test]
fn test_max_pool2d_with_indices_nan_in_window() {
    let nan = f32::NAN;
    let (values, indices, _, _) = pooled_with_indices(
        [1, 4, 4, 1],
        vec![
            1.0, nan, 3.0, 2.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, nan, //
            13.0, 14.0, 15.0, 16.0,
        ],
        MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], false),
    );
    assert!(values[0].is_nan() && values[3].is_nan(), "{values:?}");
    assert_eq!((values[1], values[2]), (8.0, 14.0));
    assert_eq!(indices, [1.0, 7.0, 13.0, 11.0]);
}

/// The same rule lane by lane: four channels, each with its NaN at another tap of a `3 × 3`
/// window, or none, against the host reference.
#[test]
fn test_max_pool2d_with_indices_nan_per_channel() {
    use cubek_pool::eval::cpu_reference::{cpu_reference_max_pool_indices, cpu_reference_pool};

    let (h, w, c) = (5, 5, 4);
    let values = (0..h * w * c)
        .map(|i| {
            let (pos, channel) = (i / c, i % c);
            match (channel, pos) {
                (0, 6) | (1, 12) | (2, 18) | (2, 24) => f32::NAN,
                _ => ((pos * 7 + channel * 3) % 11) as f32 - 5.0,
            }
        })
        .collect();
    let options = MaxPoolOptions::new([3, 3], [1, 1], [1, 1], [1, 1], false);
    let (values, indices, host, out_shape) =
        pooled_with_indices([1, h, w, c], values, options.clone());

    let problem = make_problem(Shape::from([1, h, w, c]), true, options.clone());
    let want_values = cpu_reference_pool(&host, problem);
    let want_indices = cpu_reference_max_pool_indices(&host, &options, &out_shape);
    for (i, (&got, &index)) in values.iter().zip(&indices).enumerate() {
        let at = [0, i / (w * c), i / c % w, i % c];
        let (want, want_index) = (want_values.get_f32(&at), want_indices.get_i32(&at));
        assert!(
            got == want || (got.is_nan() && want.is_nan()),
            "value at {at:?}: got {got}, want {want}"
        );
        assert_eq!(index, want_index as f32, "index at {at:?}");
    }
}
