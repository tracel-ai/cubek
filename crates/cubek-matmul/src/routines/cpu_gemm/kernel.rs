//! The CpuGemm kernel: the whole body is `c.mma(a, b)`.

use cubecl::prelude::*;
use cubek_tile::{StridedTileArg, Tile, TileSpec};

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
    // The matmul contract is `out = A·B` and `mma` accumulates, so zero first: the
    // register leaf runs in place, round-tripping K-chunk partials through `c`.
    c.zero();
    c.mma(&a, &b);
}

/// [`cpu_gemm_kernel`]'s bare-tensor twin: the identical body, operands as plain
/// tensors plus each one's comptime [`TileSpec`], tiles built in-kernel.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_bare_kernel<E: Numeric, EL: Numeric, ER: Numeric, VA: Size, VB: Size, VC: Size>(
    a: &Tensor<Vector<EL, VA>>,
    b: &Tensor<Vector<ER, VB>>,
    c: &Tensor<Vector<E, VC>>,
    #[comptime] spec_a: TileSpec,
    #[comptime] spec_b: TileSpec,
    #[comptime] spec_c: TileSpec,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
) {
    let a = Tile::<EL>::of(a, spec_a);
    let b = Tile::<ER>::of(b, spec_b);
    let mut c = Tile::<E>::of(c, spec_c);
    // The matmul contract is `out = A·B` and `mma` accumulates, so zero first: the
    // register leaf runs in place, round-tripping K-chunk partials through `c`.
    c.zero();
    c.mma(&a, &b);
}
