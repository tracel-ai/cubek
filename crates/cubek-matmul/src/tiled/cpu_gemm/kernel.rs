//! The CpuGemm kernel: the whole body is the accumulator's scope.

use cubecl::prelude::*;
use cubek_tile::{Monoid, Semiring, Space, TileArg};

/// The same three lines the tensor-core kernel runs.
///
/// Each operand arrives as one argument bundling its tensor with its comptime spec; the one
/// kernel [`Space`] is projected onto each in the first lines. `a` stays scalar (broadcast per
/// `K`); `b` and `c` carry the launch-chosen line size along their contiguous `N` axis. Each
/// keeps its own element type, `EL`/`ER` for the inputs, `EA` for the accumulator and `E` for the
/// stored output, and the leaf casts the inputs into `EA`, so mixed-precision GEMM falls out of
/// one kernel (same dtype is the `EL = ER = EA = E` case, where the casts fold away). What runs
/// at the last level is the space's own statement, so nothing about it is spelled here.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_kernel<
    E: Numeric,
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    VA: Size,
    VB: Size,
    VC: Size,
>(
    a: &TileArg<'_, EL, VA>,
    b: &TileArg<'_, ER, VB>,
    c: &TileArg<'_, E, VC>,
    #[comptime] space: Space,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(space);
    let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}
