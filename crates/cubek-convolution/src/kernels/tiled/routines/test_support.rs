//! What the routines' pure tests prepare against: devices stated as constants and the problems
//! the benchmark catalog names.

use crate::kernels::tiled::ConvProblem;
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

/// `hw` with its planes `width` units wide.
pub fn with_plane_width(mut hw: Hardware, width: u32) -> Hardware {
    let device = devices::topology(&mut hw);
    device.plane_size_min = width;
    device.plane_size_max = width;
    hw
}

/// The catalog's problem shapes, each as `(name, problem)`: few channels under a deep filter, 64
/// channels, a stride of two, and a map seven wide.
pub fn problems() -> [(&'static str, ConvProblem); 4] {
    [
        (
            "alexnet_like",
            ConvProblem::dense(8, [227, 227], (3, 64), [11, 11], 4, F16),
        ),
        (
            "conv3x3_c64",
            ConvProblem::dense(8, [58, 58], (64, 64), [3, 3], 1, F16),
        ),
        (
            "conv3x3_c64_stride2",
            ConvProblem::dense(8, [57, 57], (64, 128), [3, 3], 2, F16),
        ),
        (
            "narrow_width",
            ConvProblem::dense(8, [9, 9], (64, 64), [3, 3], 1, F16),
        ),
    ]
}
