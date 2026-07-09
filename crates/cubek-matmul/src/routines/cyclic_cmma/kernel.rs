//! The CyclicCmma kernel: promote the accumulator, contract.

use cubecl::prelude::*;
use cubek_tile::TileArg;

/// The classic global matmul, spelled in tiles: the accumulator runs the whole
/// contraction register-resident — promote it (`init_accumulator`), `c.mma(a, b)` inside
/// the bracket, and the write-back follows in `contract` (the epilogue). Each operand
/// keeps its own element type, matching the hardware's `MmaConfig`.
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
    acc.contract(&mut c, |r| r.mma(&a, &b));
}
