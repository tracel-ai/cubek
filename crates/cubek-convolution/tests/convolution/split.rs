//! The accelerated split weight gradient: the weight's gradient summed over splits of the batch and
//! the positions, each split a partial in `f32`, the partials then reduced.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_convolution::components::ConvSetupError;
use cubek_convolution::{
    AcceleratedTileKind, ConvolutionArgs,
    backward_weight::split::{AcceleratedSplitStrategy, PartialColumnOrder, launch_accelerated},
};
use cubek_matmul::definition::MatmulElems;
use cubek_std::InputBinding;
use cubek_test_utils::{ExecutionOutcome, ScopeUnderTest, TestOutcome, launch_and_capture_outcome};

use crate::convolution::{Conv, ConvRoutine, ConvTensors, Convolves, Declines, F32, Pass, Storage};

/// The split weight gradient at `0`'s strategy.
#[derive(Debug)]
struct Split(AcceleratedSplitStrategy);

impl<const N: usize> ConvRoutine<N> for Split {
    /// # Panics
    ///
    /// On a pass other than the weight's gradient, the only one it computes.
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<N>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        assert_eq!(
            pass,
            Pass::BackwardWeight,
            "the split computes the weight's gradient alone"
        );
        let read =
            |tensor: &TensorHandle| InputBinding::new(tensor.clone().binding(), conv.element());
        launch_accelerated::<N>(
            client,
            read(&tensors.input),
            read(&tensors.out_grad),
            tensors.written.clone().binding(),
            conv.args(),
            self.0,
            conv.matmul_elems(),
        )
    }
}

impl<const N: usize> Conv<N> {
    /// This convolution's weight gradient through the split on mma, its partials' columns either
    /// way, its tensors channels last and channels first.
    fn check_the_split_weight_gradient(self) {
        let convs = [
            self.stored(Storage::Dense),
            self.stored(Storage::ChannelsFirstPadded),
        ];
        for column_order in [
            PartialColumnOrder::ChannelsLast,
            PartialColumnOrder::SpatialLast,
        ] {
            let split = Convolves::new(
                Split(AcceleratedSplitStrategy {
                    tile_kind: AcceleratedTileKind::Mma,
                    column_order,
                }),
                Declines::AsItStates,
            );
            split.check(split.at(&[Pass::BackwardWeight], &convs));
        }
    }
}

#[test]
fn accelerated_split_weight_gradient_handles_tail_and_output_strides() {
    Conv::of(F32)
        .batches(8)
        .spatial([1031])
        .channels(3)
        .out_channels(7)
        .kernel([3])
        .stride([2])
        .padded([1])
        .dilation([2])
        .check_the_split_weight_gradient();
}

#[test]
fn accelerated_split_weight_gradient_supports_three_spatial_axes() {
    Conv::of(F32)
        .batches(16)
        .spatial([5, 7, 9])
        .channels(4)
        .out_channels(8)
        .kernel([3, 3, 3])
        .check_the_split_weight_gradient();
}

/// Widths a vector divides and does not, under strides, paddings and dilations that move the
/// positions a split's vectors start at.
#[test]
fn accelerated_split_weight_gradient_handles_vector_alignment() {
    for width in [20, 19] {
        for (stride, padding, dilation) in [
            ([2, 4], [0, 0], [1; 2]),
            ([2, 4], [1, 4], [1; 2]),
            ([2, 4], [1, 1], [1; 2]),
            ([2, 3], [0, 0], [1; 2]),
            ([2, 4], [0, 0], [2; 2]),
        ] {
            Conv::of(F32)
                .batches(32)
                .spatial([16, width])
                .channels(4)
                .out_channels(8)
                .kernel([4, 8])
                .stride(stride)
                .padded(padding)
                .dilation(dilation)
                .check_the_split_weight_gradient();
        }
    }
}

#[test]
fn accelerated_split_weight_gradient_keeps_f16_partials_in_f32() {
    check_reduced_precision::<half::f16>();
}

#[test]
fn accelerated_split_weight_gradient_preserves_bf16() {
    check_reduced_precision::<half::bf16>();
}

// Reads inputs [2, 512, 4] and gradients [2, 512, 8], and checks weights [8, 1, 4].
fn check_reduced_precision<E: CubeElement + Numeric + PartialEq>() {
    let client = cubecl::test_device().client();
    let dtype = E::elem_type_native();
    let mut input_values = [
        vec![E::from_int(200); 512 * 4],
        vec![E::from_int(-200); 512 * 4],
    ]
    .concat();
    for index in (512 * 4 + 3..input_values.len()).step_by(4) {
        input_values[index] = E::from_int(-199);
    }
    let grad_values = vec![E::from_int(1); 2 * 512 * 8];
    let input = TensorHandle::new(
        client.create_from_slice(E::as_bytes(&input_values)),
        [2, 512, 4],
        [512 * 4, 4, 1],
        dtype,
    );
    let grad = TensorHandle::new(
        client.create_from_slice(E::as_bytes(&grad_values)),
        [2, 512, 8],
        [512 * 8, 8, 1],
        dtype,
    );

    for column_order in [
        PartialColumnOrder::ChannelsLast,
        PartialColumnOrder::SpatialLast,
    ] {
        let output = TensorHandle::empty(&client, [8, 1, 4], dtype);
        let outcome = launch_and_capture_outcome(&client, &[&output.handle], |client| {
            let args = ConvolutionArgs {
                stride: [1],
                padding: [0],
                dilation: [1],
            };
            let input = InputBinding::new(input.clone().binding(), dtype);
            let grad = InputBinding::new(grad.clone().binding(), dtype);
            let result = launch_accelerated::<1>(
                client,
                input,
                grad,
                output.clone().binding(),
                args,
                AcceleratedSplitStrategy {
                    tile_kind: AcceleratedTileKind::Mma,
                    column_order,
                },
                MatmulElems::from_single_dtype(dtype),
            );
            match result {
                Ok(()) => ExecutionOutcome::Executed,
                Err(error) => ExecutionOutcome::CompileError(format!("{error:?}")),
            }
        });
        if let ExecutionOutcome::CompileError(error) = outcome {
            TestOutcome::CompileError(error).enforce();
            return;
        }
        let strides = output.strides().to_vec();
        let bytes = client.read_one_unchecked(output.handle);
        let actual = E::from_bytes(&bytes);
        for co in 0..8 {
            for ci in 0..4 {
                let expected = if ci == 3 { 512 } else { 0 };
                assert_eq!(
                    actual[co * strides[0] + ci * strides[2]],
                    E::from_int(expected)
                );
            }
        }
    }
}

#[test]
fn accelerated_split_weight_gradient_rejects_wide_source_coordinates() {
    let client = cubecl::test_device().client();
    let conv = Conv::<2>::of(F32);
    let input = conv.store(&client, &[1, 1, 1, 4], &[1.0; 4], Storage::Dense);
    let grad = conv.store(&client, &[1, 1, 1, 8], &[1.0; 8], Storage::Dense);
    let output = conv.store(&client, &[8, 3, 3, 4], &[0.0; 288], Storage::Dense);
    let binding = |tensor: &TensorHandle| InputBinding::new(tensor.clone().binding(), F32);
    let result = launch_accelerated::<2>(
        &client,
        binding(&input),
        binding(&grad),
        output.binding(),
        ConvolutionArgs {
            stride: [1; 2],
            padding: [0; 2],
            dilation: [1usize << 31; 2],
        },
        AcceleratedSplitStrategy {
            tile_kind: AcceleratedTileKind::Mma,
            column_order: PartialColumnOrder::ChannelsLast,
        },
        MatmulElems::from_single_dtype(f32::elem_type_native()),
    );
    assert!(matches!(result, Err(ConvSetupError::InvalidConfig(_))));
}
