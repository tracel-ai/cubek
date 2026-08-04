//! The CpuGemm kernel: the whole body is `c.mma(a, b)`.

use cubecl::prelude::*;
use cubek_tile::{Space, TileArg};

/// The whole body is `c.mma(a, b)`. Each operand arrives as one argument bundling its
/// tensor with its comptime spec; the one kernel [`Space`] is projected onto each in the
/// first lines. `a` stays scalar (broadcast per `K`); `b` and `c` carry the
/// launch-chosen line size along their contiguous `N` axis. Each keeps its own element
/// type, `EL`/`ER` for the inputs, `E` for the accumulator/output, and the leaf casts
/// the inputs into `E`, so mixed-precision GEMM falls out of one kernel (same dtype is
/// the `EL = ER = E` case, where the casts fold away).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_kernel<E: Numeric, EL: Numeric, ER: Numeric, VA: Size, VB: Size, VC: Size>(
    a: &TileArg<'_, EL, VA>,
    b: &TileArg<'_, ER, VB>,
    c: &TileArg<'_, E, VC>,
    #[comptime] space: Space,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    // The matmul contract is `out = A·B` and `mma` accumulates, so zero first: the
    // register leaf runs in place, round-tripping K-chunk partials through `c`.
    c.zero();
    c.mma(&a, &b);
}
