//! The CyclicCmma kernel: the whole body is `c.mma(a, b)`.

use cubecl::prelude::*;
use cubek_tile::TileArg;

/// The whole body is `c.mma(a, b)`; the space's `Leaf::Cmma` routes the lowering through
/// the resident-fragment boundary hoist and the tensor-core leaf. Each operand keeps its
/// own element type, matching the hardware's `MmaConfig`.
#[cube(launch)]
pub fn cyclic_cmma_kernel<E: Numeric, EL: Numeric, ER: Numeric>(
    a: &TileArg<'_, EL>,
    b: &TileArg<'_, ER>,
    c: &TileArg<'_, E>,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}
