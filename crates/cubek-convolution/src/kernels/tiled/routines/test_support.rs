//! What the routines' pure tests prepare against: devices stated as constants and the problems
//! the benchmark catalog names.

use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem};
use crate::launch::ConvolutionArgs;
use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubek_std::{Hardware, devices};

pub const F16: ElemType = ElemType::Float(FloatKind::F16);
pub const F32: ElemType = ElemType::Float(FloatKind::F32);

fn shape(element: ElemType, (m, n, k): (u32, u32, u32)) -> MmaConfig {
    MmaConfig {
        a_type: element,
        b_type: element,
        cd_type: F32,
        m,
        n,
        k,
    }
}

/// An NVIDIA GPU offering cmma `16x16x16` and `8x32x16`, and manual mma `16x8x8`, all over `f16`.
pub fn nvidia() -> Hardware {
    let hw = devices::offering_cmma(
        devices::h100(),
        [shape(F16, (16, 16, 16)), shape(F16, (8, 32, 16))],
    );
    devices::offering_mma(hw, [shape(F16, (16, 8, 8))])
}

/// A device offering the cmma instruction alone.
pub fn cmma_only() -> Hardware {
    devices::offering_cmma(devices::h100(), [shape(F16, (16, 16, 16))])
}

/// A device offering the cmma instruction of eight rows alone.
pub fn cmma_8x32_only() -> Hardware {
    devices::offering_cmma(devices::h100(), [shape(F16, (8, 32, 16))])
}

/// A dense NHWC forward convolution with no padding or dilation, in `f16`: the 64 channel layer of
/// a 58 wide map the tests share. Tests change the fields they study and state the problem the
/// launch sees through `ConvProblem::from`, which derives each operand's alignment from its role.
pub fn dense() -> Conv2dProblem {
    Conv2dProblem {
        input_shape: [8, 64, 58, 58],
        weight_shape: [64, 64, 3, 3],
        args: ConvolutionArgs {
            stride: [1; 2],
            padding: [0; 2],
            dilation: [1; 2],
        },
        operation: ConvolutionOperation::Forward,
        element: Conv2dElement::F16,
    }
}

/// The problems the pure tests prepare, each as `(name, problem)`: few channels under a deep
/// filter, 64 channels, and a map seven wide.
pub fn problems() -> [(&'static str, Conv2dProblem); 3] {
    [
        (
            "few_channels_deep_filter",
            Conv2dProblem {
                input_shape: [8, 3, 227, 227],
                weight_shape: [64, 3, 11, 11],
                args: ConvolutionArgs {
                    stride: [4; 2],
                    ..dense().args
                },
                ..dense()
            },
        ),
        ("channels_64", dense()),
        (
            "map_7_wide",
            Conv2dProblem {
                input_shape: [8, 64, 9, 9],
                ..dense()
            },
        ),
    ]
}
