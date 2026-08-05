//! The per-tensor scale of a two-level scheme.
//!
//! Reading is split from applying so a kernel scaling several blocks per unit can hoist the load
//! out of its loop. `FG` is the scale's own type, not the compute type: reading an f32 scale as
//! f16 because that is what the kernel computes in returns garbage.
//!
//! The effective scale stays in f32 all the way to the multiply. It is the product of a tensor
//! magnitude and a block spread, so it can be subnormal in a narrow compute type even when every
//! value it scales is representable there.

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
pub(crate) fn apply_global<FG: Numeric, FS: CubePrimitive>(
    block: FS,
    global: ComptimeOption<FG>,
) -> f32 {
    #[comptime]
    match global {
        ComptimeOption::Some(global) => f32::cast_from(global) * f32::cast_from(block),
        ComptimeOption::None => f32::cast_from(block),
    }
}
