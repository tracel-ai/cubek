//! The CpuGemm kernel: the whole body is `c.mma(a, b)`.

use cubecl::prelude::*;
use cubek_tile::StridedTileArg;

/// The whole body is `c.mma(a, b)`. `a` stays scalar (broadcast per `K`); `b` and `c` carry the
/// launch-chosen line size along their contiguous `N` axis (set on each operand's builder). Each keeps its own
/// element type — `EL`/`ER` for the inputs, `E` for the accumulator/output — and the leaf casts
/// the inputs into `E`, so mixed-precision GEMM falls out of one kernel (same dtype is the
/// `EL = ER = E` case, where the casts fold away).
#[cube(launch)]
pub fn cpu_gemm_kernel<E: Numeric, EL: Numeric, ER: Numeric>(
    a: &StridedTileArg<'_, EL>,
    b: &StridedTileArg<'_, ER>,
    c: &StridedTileArg<'_, E>,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}
