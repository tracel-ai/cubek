//! The Cmma kernel: promote the accumulator, contract, copy back.

use cubecl::prelude::*;
use cubek_tile::{DeliveryFamily, LeafOp, Space, TileArg};

/// The classic global matmul, spelled in tiles, one body for both delivery families
/// (strided cooperative copy or TMA bulk copy; the output is always strided). Each
/// operand arrives as one argument bundling its tensor with its comptime spec, and the
/// one kernel [`Space`] is projected onto each in the first lines. Each operand keeps
/// its own element type, matching the hardware's `MmaConfig`; the accumulator is
/// resident in `EA` (typically `f32`) and cast down to the output `E` on drain.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cmma_kernel<
    E: Numeric,
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    VA: Size,
    VB: Size,
    VC: Size,
    D: DeliveryFamily,
>(
    a: &D::Arg<EL, VA>,
    b: &D::Arg<ER, VB>,
    c: &TileArg<'_, E, VC>,
    #[comptime] space: Space,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let a = D::tile::<EL, VA>(a, comptime!(space.clone()));
    let b = D::tile::<ER, VB>(b, comptime!(space.clone()));
    let c = c.tile(space);
    let mut acc = c.accumulate::<EA, _>(&a, LeafOp::Sum);
    // The matmul contract is `out = A·B` and `mma` accumulates, so start at zero.
    acc.zero();
    // The contraction exhausts the scope `out` opened, so this drains it too.
    acc.mma(&a, &b);
}
