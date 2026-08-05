//! The per-tensor scale of a two-level scheme.
//!
//! Reading is split from applying so a kernel scaling several blocks per unit can hoist the load
//! out of its loop. `FG` is the scale's own type, not the compute type: reading an f32 scale as
//! f16 because that is what the kernel computes in returns garbage.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::LinearView;

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
        // The product is in range even when neither factor is: the scheme puts the tensor's
        // magnitude in one and the spread in the other. Casting them separately to a narrow `F`
        // flushes the per-tensor scale to zero.
        ComptimeOption::Some(global) => {
            F::cast_from(f32::cast_from(global) * f32::cast_from(block))
        }
        ComptimeOption::None => F::cast_from(block),
    }
}
