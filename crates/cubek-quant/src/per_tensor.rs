//! The per-tensor scale of a two-level scheme.
//!
//! Every helper here takes the scale as a [ComptimeOption], so a one-level scheme emits none of
//! this: no load, no multiply, no writeback. Reading is split from applying so that a kernel
//! scaling several blocks per unit can hoist the load out of its loop and keep the scale in a
//! register.
//!
//! The scale has its own element type `FG`, taken from [`QuantLevel::BlockTensor`]'s `global`
//! field. It is not the compute type: reading an f32 scale as f16 because that is what the kernel
//! computes in returns garbage.

use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

/// Load the per-tensor scale, once.
#[cube]
pub(crate) fn read_global<FG: Numeric>(
    global: ComptimeOption<LinearView<'_, FG>>,
) -> ComptimeOption<FG> {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => ComptimeOption::new_Some(global.read(0)),
        ComptimeOption::None => ComptimeOption::new_None(),
    }
}

/// The scale to work against: the block scale, times the per-tensor scale when there is one.
#[cube]
pub(crate) fn apply_global<F: Float, FG: Numeric, FS: CubePrimitive>(
    block: FS,
    global: ComptimeOption<FG>,
) -> F {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => F::cast_from(global) * F::cast_from(block),
        ComptimeOption::None => F::cast_from(block),
    }
}

/// Copy the per-tensor scale into the quantized tensor's own scale region, where dequantize reads
/// it back from.
#[cube]
pub(crate) fn write_global<FG: Numeric>(
    global: ComptimeOption<FG>,
    out_global: ComptimeOption<LinearViewMut<'_, FG>>,
) {
    #[comptime]
    match out_global {
        ComptimeOption::Some(mut out) =>
        {
            #[comptime]
            match global {
                ComptimeOption::Some(global) => {
                    if ABSOLUTE_POS == 0 {
                        out.write(0, global);
                    }
                }
                ComptimeOption::None => {}
            }
        }
        ComptimeOption::None => {}
    }
}
