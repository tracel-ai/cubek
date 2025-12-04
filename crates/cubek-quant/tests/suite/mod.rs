use cubecl::Runtime;
use cubecl::prelude::*;
use cubek_quant::scheme::{QuantLevel, QuantParam};

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
