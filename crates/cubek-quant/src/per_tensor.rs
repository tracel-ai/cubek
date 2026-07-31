//! The per-tensor scale of a two-level scheme.
//!
//! Reading is split from applying so a kernel scaling several blocks per unit can hoist the load
//! out of its loop. `FG` is the scale's own type, not the compute type: reading an f32 scale as
//! f16 because that is what the kernel computes in returns garbage.

use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

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
