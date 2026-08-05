use cubecl::{
    Runtime, TestRuntime, client::ComputeClient, prelude::*, std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_quant::scheme::{QuantLevel, QuantParam};
mod scale_rounding;
mod tiled;
mod two_level_contract;
mod ue4m3_two_level;

/// `data` on the device as an f32 tensor: what these tests hand the kernels for an input, a block
/// scale grid, or a one-element per-tensor scale.
pub(crate) fn f32_tensor(
    client: &ComputeClient<TestRuntime>,
    data: &[f32],
    shape: Shape,
) -> TensorHandle<TestRuntime> {
    let alloc =
        client.create_tensor_from_slice(f32::as_bytes(data), shape.clone(), f32::type_size());

    TensorHandle::new(
        alloc.memory,
        shape,
        alloc.strides,
        f32::as_type_native_unchecked(),
    )
}

#[macro_export]
macro_rules! testgen_quant {
    ($value: expr, $shape_x: expr, $shape_y: expr) => {
        static SHAPE_X: usize = $shape_x;
        static SHAPE_Y: usize = $shape_y;
        static VALUE: QuantValue = $value;

        include!("symmetric.rs");
    };

    ($shape_x: expr, $shape_y: expr) => {
        mod q8f {
            use super::*;
            testgen_quant!(QuantValue::Q8F, $shape_x, $shape_y);
        }
        mod q8s {
            use super::*;
            testgen_quant!(QuantValue::Q8S, $shape_x, $shape_y);
        }
        mod q4f {
            use super::*;
            testgen_quant!(QuantValue::Q4F, $shape_x, $shape_y);
        }
        mod q4s {
            use super::*;
            testgen_quant!(QuantValue::Q4S, $shape_x, $shape_y);
        }
        mod q2f {
            use super::*;
            testgen_quant!(QuantValue::Q2F, $shape_x, $shape_y);
        }
        mod q2s {
            use super::*;
            testgen_quant!(QuantValue::Q2S, $shape_x, $shape_y);
        }
    };
    () => {
        mod quant {
            use super::*;

            mod size32x32 {
                use super::*;
                testgen_quant!(32, 32);
            }
            mod size16x64 {
                use super::*;
                testgen_quant!(16, 64);
            }
        }
    };
}

testgen_quant!();
