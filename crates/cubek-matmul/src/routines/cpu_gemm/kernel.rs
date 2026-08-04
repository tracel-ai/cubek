//! The CpuGemm kernel: the whole body is `c.mma(a, b)`.

use cubecl::prelude::*;
use cubek_tile::{Space, Tile, TileSpec};

/// The whole body is `c.mma(a, b)`. Each operand is a plain tensor plus its comptime
/// [`TileSpec`]; the one kernel [`Space`] is projected onto each operand's axes in the
/// first lines. `a` stays scalar
/// (broadcast per `K`); `b` and `c` carry the launch-chosen line size along their
/// contiguous `N` axis. Each keeps its own element type — `EL`/`ER` for the inputs, `E`
/// for the accumulator/output — and the leaf casts the inputs into `E`, so
/// mixed-precision GEMM falls out of one kernel (same dtype is the `EL = ER = E` case,
/// where the casts fold away).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_kernel<E: Numeric, EL: Numeric, ER: Numeric, VA: Size, VB: Size, VC: Size>(
    a: &Tensor<Vector<EL, VA>>,
    b: &Tensor<Vector<ER, VB>>,
    c: &Tensor<Vector<E, VC>>,
    #[comptime] space: Space,
    #[comptime] spec_a: TileSpec,
    #[comptime] spec_b: TileSpec,
    #[comptime] spec_c: TileSpec,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
) {
    let a = Tile::<EL>::of(a, comptime!(space.clone()), spec_a);
    let b = Tile::<ER>::of(b, comptime!(space.clone()), spec_b);
    let mut c = Tile::<E>::of(c, space, spec_c);
    // The matmul contract is `out = A·B` and `mma` accumulates, so zero first: the
    // register leaf runs in place, round-tripping K-chunk partials through `c`.
    c.zero();
    c.mma(&a, &b);
}
