//! The tensors that name no convolution the tiled kernels compute, refused before any kernel
//! runs: each case states the operation, its shapes and window, and the [`ConvDecline`] it earns.

use super::ConvTensors;
use crate::components::ConvolutionOperation;
use crate::kernels::tiled::ConvDecline;
use crate::launch::SpatialWindow;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::zspace::Shape;
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
    forward_weight_of_other_channels: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 16], [1, 6, 6, 32]], SAME => ConvDecline::Groups;
    forward_output_of_other_channels: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 16]], SAME => ConvDecline::OutChannels;
    forward_output_of_another_batch: Forward, false, [[2, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], SAME => ConvDecline::Batch;
    forward_output_of_another_extent: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 5, 6, 32]], SAME => ConvDecline::ForwardExtent;
    forward_weight_of_no_taps: Forward, false, [[1, 6, 6, 64], [32, 0, 3, 64], [1, 6, 6, 32]], SAME => ConvDecline::KernelEmpty;
    forward_input_of_no_channels: Forward, false, [[1, 6, 6, 0], [32, 3, 3, 0], [1, 6, 6, 32]], SAME => ConvDecline::ContractionEmpty;
    forward_stride_of_zero: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], [window(0, 1, 1), window(1, 1, 1)] => ConvDecline::WindowStepZero;
    forward_dilation_of_zero: Forward, false, [[1, 6, 6, 64], [32, 3, 3, 64], [1, 6, 6, 32]], [window(1, 1, 1), window(1, 1, 0)] => ConvDecline::WindowStepZero;

    weight_grad_quantized_input: BackwardWeight, true, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::Quantized;
    weight_grad_of_other_channels: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 16]], SAME => ConvDecline::Groups;
    weight_grad_of_other_filters: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [16, 3, 3, 64]], SAME => ConvDecline::OutChannels;
    weight_grad_out_grad_of_another_batch: BackwardWeight, false, [[2, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::Batch;
    weight_grad_out_grad_of_another_extent: BackwardWeight, false, [[1, 6, 6, 64], [1, 5, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::WeightGradExtent;
    weight_grad_of_no_taps: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 0, 3, 64]], SAME => ConvDecline::KernelEmpty;
    weight_grad_over_an_empty_batch: BackwardWeight, false, [[0, 6, 6, 64], [0, 6, 6, 32], [32, 3, 3, 64]], SAME => ConvDecline::ContractionEmpty;
    weight_grad_stride_of_zero: BackwardWeight, false, [[1, 6, 6, 64], [1, 6, 6, 32], [32, 3, 3, 64]], [window(1, 1, 1), window(0, 1, 1)] => ConvDecline::WindowStepZero;

    data_grad_strided: BackwardData, false, [[1, 3, 3, 32], [32, 3, 3, 64], [1, 6, 6, 64]], STRIDE_2 => ConvDecline::DataBackwardStrided;
    data_grad_quantized_out_grad: BackwardData, true, [[1, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::Quantized;
    data_grad_of_other_channels: BackwardData, false, [[1, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 16]], SAME => ConvDecline::Groups;
    data_grad_of_other_filters: BackwardData, false, [[1, 6, 6, 32], [16, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::OutChannels;
    data_grad_out_grad_of_another_batch: BackwardData, false, [[2, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::Batch;
    data_grad_out_grad_of_another_extent: BackwardData, false, [[1, 5, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::DataGradExtent;
    data_grad_weights_of_no_taps: BackwardData, false, [[1, 6, 6, 32], [32, 0, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::KernelEmpty;
    data_grad_of_no_filters: BackwardData, false, [[1, 6, 6, 0], [0, 3, 3, 64], [1, 6, 6, 64]], SAME => ConvDecline::ContractionEmpty;
    data_grad_dilation_of_zero: BackwardData, false, [[1, 6, 6, 32], [32, 3, 3, 64], [1, 6, 6, 64]], [window(1, 1, 0), window(1, 1, 1)] => ConvDecline::WindowStepZero;
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
