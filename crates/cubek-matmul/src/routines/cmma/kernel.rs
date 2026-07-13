//! The Cmma kernel: promote the accumulator, contract, copy back.

use cubecl::prelude::*;
use cubek_tile::{DeliveryFamily, StridedTileArg};

/// The classic global matmul, spelled in tiles, one body for both delivery families
/// (strided cooperative copy or TMA bulk copy; the output is always strided). Each
/// operand keeps its own element type, matching the hardware's `MmaConfig`.
#[cube(launch)]
pub fn cmma_kernel<E: Numeric, EL: Numeric, ER: Numeric, D: DeliveryFamily>(
    a: &D::Arg<EL>,
    b: &D::Arg<ER>,
    c: &StridedTileArg<'_, E>,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
) {
    let a = D::tile::<EL>(a);
    let b = D::tile::<ER>(b);
    let mut c = c.tile();
    let mut acc = c.promote();
    acc.mma(&a, &b);
    c.copy_from(&acc);
}
