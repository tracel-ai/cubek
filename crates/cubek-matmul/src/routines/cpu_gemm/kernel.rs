//! The CpuGemm kernel: the whole body is `c.mma(a, b)`.

use cubecl::prelude::*;
use cubek_tile::TileArg;

/// The whole body is `c.mma(a, b)`. `a` stays scalar (broadcast per `K`); `b` and `c`
/// carry the launch-chosen line size `V` along their contiguous `N` axis, so the leaf
/// contraction reads genuine `Vector<E, V>` lines.
#[cube(launch)]
pub fn cpu_gemm_kernel<E: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[define(E)] _dtype: StorageType,
    #[define(V)] _vector_size: usize,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}
