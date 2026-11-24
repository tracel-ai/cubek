//! The tensors that name no convolution the tiled kernels compute, refused before any kernel
//! runs: each case states the operation, its shapes and window, and the [`ConvDecline`] it earns.

use super::ConvTensors;
use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{ChannelAxis, ConvDecline, ConvGrouping, SpatialDim};
use crate::launch::SpatialWindow;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::zspace::{Shape, Tiling};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::InputBinding;

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

const fn window(stride: usize, padding: usize, dilation: usize) -> SpatialWindow {
    SpatialWindow {
        stride,
        padding,
        dilation,
    }
}

const SAME: [SpatialWindow; 2] = [window(1, 1, 1); 2];
const STRIDE_2: [SpatialWindow; 2] = [window(2, 1, 1); 2];

/// A launch of dense half precision operands to refuse, shaped as stated; `quantized` makes the
/// lhs a quantized one, which is declined whatever else the launch states.
struct Refused {
    operation: ConvolutionOperation,
    quantized: bool,
    shapes: Vec<Vec<usize>>,
    window: [SpatialWindow; 2],
}

impl Refused {
    fn new(
        operation: ConvolutionOperation,
        quantized: bool,
        shapes: [[usize; 4]; 3],
        window: [SpatialWindow; 2],
    ) -> Self {
        Self {
            operation,
            quantized,
            shapes: shapes.map(|shape| shape.to_vec()).to_vec(),
            window,
        }
    }

    /// The decline the tensors are refused with.
    fn declined(&self) -> Decline {
        let client = cubecl::test_device().client();
        let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: F16,
            rhs: F16,
            out: F16,
        });
        let dense = |shape: &[usize]| TensorHandle::empty(&client, shape.to_vec(), F16).binding();
        let [lhs, rhs, out] = [0, 1, 2].map(|at| dense(&self.shapes[at]));
        let lhs = match self.quantized {
            false => InputBinding::new(lhs, F16),
            true => InputBinding::Quantized {
                scale: lhs.clone(),
                shape: Shape::from(self.shapes[0].clone()),
                data: lhs,
                data_dtype: F16,
                scale_dtype: F32,
                scheme: Default::default(),
            },
        };
        let rhs = InputBinding::new(rhs, F16);
        let window = &self.window;
        let tensors = match self.operation {
            ConvolutionOperation::Forward => {
                ConvTensors::forward(&client, lhs, rhs, None, out, window, dtypes)
            }
            ConvolutionOperation::BackwardWeight => {
                ConvTensors::backward_weight(&client, lhs, rhs, out, window, dtypes)
            }
            ConvolutionOperation::BackwardData => {
                ConvTensors::backward_data(&client, lhs, rhs, out, window, dtypes)
            }
            ConvolutionOperation::ForwardTransposed => {
                unreachable!("no refusal row states a transposed forward")
            }
        };
        match tensors {
            Ok(_) => panic!("the tensors were accepted"),
            Err(declined) => declined,
        }
    }
}

/// A test for each launch that must be refused, `name: operation, quantized, shapes, window =>
/// refusal`, the shapes as `[lhs, rhs, out]`.
macro_rules! refusals {
    ($($name:ident: $operation:ident, $quantized:expr, $shapes:expr, $window:expr => $declined:expr;)*) => {
        $(#[test]
        fn $name() {
            let launch = Refused::new(ConvolutionOperation::$operation, $quantized, $shapes, $window);
            assert_eq!(launch.declined(), $declined.into());
        })*
    };
}

refusals! {
    forward_quantized_input: Forward, true, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], SAME => ConvDecline::Quantized;
    forward_weight_of_channels_no_group_count_states: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 24], [1, 6, 6, 32]], SAME => ConvDecline::Groups;
    forward_filters_no_group_count_splits: Forward, false, [[1, 6, 6, 64], [30, 3, 3, 16], [1, 6, 6, 30]], SAME => ConvDecline::Groups;
    forward_output_channels_no_group_count_splits: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 16], [1, 6, 6, 30]], SAME => ConvDecline::OutChannels;
    forward_depthwise: Forward, false, [[1, 6, 6, 16], [16, 3, 3, 1], [1, 6, 6, 16]], SAME => ConvDecline::Depthwise;
    forward_output_of_other_channels: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 16]], SAME => ConvDecline::OutChannels;
    forward_output_of_another_batch: Forward, false, [[2, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], SAME => ConvDecline::Batch;
    forward_output_of_another_extent: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 5, 6, 32]], SAME => ConvDecline::ForwardExtent;
    forward_weight_of_no_taps: Forward, false, [[1, 6, 6, 64], [32, 0, 3, 64], [1, 6, 6, 32]], SAME => ConvDecline::KernelEmpty;
    forward_input_of_no_channels: Forward, false, [[1, 6, 6, 0], [32, 3, 3, 0], [1, 6, 6, 32]], SAME => ConvDecline::ContractionEmpty;
    forward_stride_of_zero: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], [window(0, 1, 1), window(1, 1, 1)] => ConvDecline::WindowStepZero;
    forward_dilation_of_zero: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], [window(1, 1, 1), window(1, 1, 0)] => ConvDecline::WindowStepZero;

    weight_grad_quantized_input: BackwardWeight, true, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::Quantized;
    weight_grad_of_channels_no_group_count_states: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 24]], SAME => ConvDecline::Groups;
    weight_grad_filters_no_group_count_splits: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 30], [30, 3, 3, 16]], SAME => ConvDecline::Groups;
    weight_grad_depthwise: BackwardWeight, false, [[1, 6, 6, 16], [1, 6, 6, 16], [16, 3, 3, 1]], SAME => ConvDecline::Depthwise;
    weight_grad_of_other_filters: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [16, 3, 3, 64]], SAME => ConvDecline::OutChannels;
    weight_grad_out_grad_of_another_batch: BackwardWeight, false, [[2, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::Batch;
    weight_grad_out_grad_of_another_extent: BackwardWeight, false, [[1, 6, 6, 64], [1, 5, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::WeightGradExtent;
    weight_grad_of_no_taps: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 0, 3, 64]], SAME => ConvDecline::KernelEmpty;
    weight_grad_over_an_empty_batch: BackwardWeight, false, [[0, 6, 6, 64], [0, 6, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::ContractionEmpty;
    weight_grad_stride_of_zero: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 64]], [window(1, 1, 1), window(0, 1, 1)] => ConvDecline::WindowStepZero;

    data_grad_strided_dilated: BackwardData, false, [[1, 3, 3, 32], [32, 3, 3, 64], [1, 6, 6, 64]], [window(2, 1, 2), window(1, 1, 1)] => ConvDecline::DataBackwardStridedDilated;
    data_grad_quantized_out_grad: BackwardData, true, [[1, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::Quantized;
    data_grad_of_other_channels: BackwardData, false, [[1, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 16]], SAME => ConvDecline::Groups;
    data_grad_of_channels_no_group_count_states: BackwardData, false, [[1, 6, 6, 32], [32, 3, 3, 24], [1, 6, 6, 64]], SAME => ConvDecline::Groups;
    data_grad_filters_no_group_count_splits: BackwardData, false, [[1, 6, 6, 30], [30, 3, 3, 16], [1, 6, 6, 64]], SAME => ConvDecline::Groups;
    data_grad_depthwise: BackwardData, false, [[1, 6, 6, 16], [16, 3, 3, 1], [1, 6, 6, 16]], SAME => ConvDecline::Depthwise;
    data_grad_of_other_filters: BackwardData, false, [[1, 6, 6, 32], [16, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::OutChannels;
    data_grad_out_grad_of_another_batch: BackwardData, false, [[2, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::Batch;
    data_grad_out_grad_of_another_extent: BackwardData, false, [[1, 5, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::DataGradExtent;
    data_grad_weights_of_no_taps: BackwardData, false, [[1, 6, 6, 32], [32, 0, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::KernelEmpty;
    data_grad_of_no_filters: BackwardData, false, [[1, 6, 6, 0], [0, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::ContractionEmpty;
    data_grad_dilation_of_zero: BackwardData, false, [[1, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], [window(1, 1, 0), window(1, 1, 1)] => ConvDecline::WindowStepZero;
}

/// A strided data backward without a dilation states one convolution, in the output gradient's own
/// extents.
#[test]
fn a_strided_data_backward_is_served() {
    let client = cubecl::test_device().client();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: F16,
        rhs: F16,
        out: F16,
    });
    let dense = |shape: [usize; 4]| -> TensorBinding {
        TensorHandle::empty(&client, shape.to_vec(), F16).binding()
    };
    let tensors = ConvTensors::backward_data(
        &client,
        InputBinding::new(dense([1, 3, 3, 32]), F16),
        InputBinding::new(dense([32, 3, 3, 64]), F16),
        dense([1, 6, 6, 64]),
        &STRIDE_2,
        dtypes,
    )
    .expect("a strided data backward is served");
    let problem = tensors.problem();
    let extents = |dim| {
        let spatial_dim: &SpatialDim = &problem.spatial[dim];
        (spatial_dim.phased_out(), spatial_dim.phased_taps())
    };
    assert_eq!(extents(0), (4, 2));
    assert_eq!(extents(1), (4, 2));
}

/// Tensors of one more dim than the window gives are refused by name; `Refused` states rank 4
/// shapes, so this launch is its own.
#[test]
fn tensors_of_another_rank_than_the_window_are_refused() {
    let client = cubecl::test_device().client();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: F16,
        rhs: F16,
        out: F16,
    });
    let dense = |shape: [usize; 5]| -> TensorBinding {
        TensorHandle::empty(&client, shape.to_vec(), F16).binding()
    };
    let declined = ConvTensors::forward(
        &client,
        InputBinding::new(dense([1, 6, 6, 6, 64]), F16),
        InputBinding::new(dense([32, 3, 3, 3, 64]), F16),
        None,
        dense([1, 6, 6, 6, 32]),
        &SAME,
        dtypes,
    )
    .expect_err("the tensors were accepted");
    assert_eq!(declined, ConvDecline::RankMismatch.into());
}

/// A tiled tensor cannot have its channels split by group, which the tiling labels the dims of.
#[test]
fn a_tiled_tensor_of_a_grouped_convolution_is_refused() {
    let client = cubecl::test_device().client();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: F16,
        rhs: F16,
        out: F16,
    });
    let dense = |shape: [usize; 4]| -> TensorBinding {
        TensorHandle::empty(&client, shape.to_vec(), F16).binding()
    };
    let tiled = dense([1, 6, 6, 64])
        .with_tiling(Tiling::new(&[2]).expect("a tiling of one dim"))
        .expect("a tiling that fits the rank");
    let declined = ConvTensors::forward(
        &client,
        InputBinding::new(tiled, F16),
        InputBinding::new(dense([32, 3, 3, 16]), F16),
        None,
        dense([1, 6, 6, 32]),
        &SAME,
        dtypes,
    )
    .expect_err("the tensors were accepted");
    assert_eq!(declined, ConvDecline::PermutedTiled.into());
}

/// A forward of 32 filters whose bias has `shape` and `element`, the sum's by default.
fn forward_with_bias(shape: [usize; 1], element: Option<ElemType>) -> Decline {
    let client = cubecl::test_device().client();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: F16,
        rhs: F16,
        out: F16,
    });
    let element = element.unwrap_or(dtypes.acc_global);
    let dense = |shape: &[usize], element| -> TensorBinding {
        TensorHandle::empty(&client, shape.to_vec(), element).binding()
    };
    ConvTensors::forward(
        &client,
        InputBinding::new(dense(&[1, 6, 6, 64], F16), F16),
        InputBinding::new(dense(&[32, 3, 3, 64], F16), F16),
        Some(InputBinding::new(dense(&shape, element), element)),
        dense(&[1, 6, 6, 32], F16),
        &SAME,
        dtypes,
    )
    .expect_err("the tensors were accepted")
}

/// A bias is one value for each output channel.
#[test]
fn a_bias_of_fewer_values_than_the_output_channels_is_refused() {
    assert_eq!(forward_with_bias([31], None), ConvDecline::BiasShape.into());
}

/// A bias is of the element the output channels are summed in.
#[test]
fn a_bias_of_another_element_than_the_sum_is_refused() {
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: F16,
        rhs: F16,
        out: F16,
    });
    let other = match dtypes.acc_global == F32 {
        true => F16,
        false => F32,
    };
    assert_eq!(
        forward_with_bias([32], Some(other)),
        ConvDecline::BiasElement.into()
    );
}

#[test]
fn the_channel_axis_splits_by_shape_and_strides() {
    let client = cubecl::test_device().client();
    let dense = |shape: [usize; 4]| -> TensorBinding {
        TensorHandle::empty(&client, shape.to_vec(), F16).binding()
    };
    let grouping = ConvGrouping::of(4);

    let activation = dense([1, 6, 6, 64]);
    let [n, h, w, c] = [0, 1, 2, 3].map(|dim| activation.strides[dim]);
    let split =
        super::axes::split_channels(ChannelAxis::Last, activation, grouping).expect("untiled");
    assert_eq!(split.shape.to_vec(), [1, 6, 6, 4, 16]);
    assert_eq!(split.strides.to_vec(), [n, h, w, c * 16, c]);

    let weight = dense([32, 3, 3, 16]);
    let [f, kh, kw, c] = [0, 1, 2, 3].map(|dim| weight.strides[dim]);
    let split = super::axes::split_channels(ChannelAxis::First, weight, grouping).expect("untiled");
    assert_eq!(split.shape.to_vec(), [4, 8, 3, 3, 16]);
    assert_eq!(split.strides.to_vec(), [f * 8, f, kh, kw, c]);
}
