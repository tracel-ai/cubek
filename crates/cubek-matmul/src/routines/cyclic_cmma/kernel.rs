//! The CyclicCmma kernel: promote the accumulator, contract, copy back.

use cubecl::prelude::*;
use cubek_tile::TileArg;

/// The classic global matmul, spelled in tiles. Each operand keeps its own element
/// type, matching the hardware's `MmaConfig`.
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
    let mut acc = c.promote();
    acc.mma(&a, &b);
    c.copy_from(&acc);
}
