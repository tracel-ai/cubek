//! The per-tensor scale of a two-level scheme.
//!
//! Every helper here takes the scale as a [ComptimeOption], so a one-level scheme emits none of
//! this: no load, no multiply, no writeback. Reading is split from applying so that a kernel
//! scaling several blocks per unit can hoist the load out of its loop and keep the scale in a
//! register.

use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

/// Load the per-tensor scale, once.
#[cube]
pub(crate) fn read_global<F: Float>(
    global: ComptimeOption<LinearView<'_, F>>,
) -> ComptimeOption<F> {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => ComptimeOption::new_Some(global.read(0)),
        ComptimeOption::None => ComptimeOption::new_None(),
    }
}

/// The scale to work against: the block scale, times the per-tensor scale when there is one.
#[cube]
pub(crate) fn apply_global<F: Float, FS: CubePrimitive>(block: FS, global: ComptimeOption<F>) -> F {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => global * F::cast_from(block),
        ComptimeOption::None => F::cast_from(block),
    }
}

/// Copy the per-tensor scale into the quantized tensor's own scale region, where dequantize reads
/// it back from.
#[cube]
pub(crate) fn write_global<F: Float>(
    global: ComptimeOption<F>,
    out_global: ComptimeOption<LinearViewMut<'_, F>>,
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
