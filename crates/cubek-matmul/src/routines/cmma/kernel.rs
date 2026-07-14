//! The Cmma kernel: promote the accumulator, contract, copy back.

use cubecl::prelude::*;
use cubek_tile::{DeliveryFamily, StridedTileArg};

/// The classic global matmul, spelled in tiles, one body for both delivery families
/// (strided cooperative copy or TMA bulk copy; the output is always strided). Each
/// operand keeps its own element type, matching the hardware's `MmaConfig`; the
/// accumulator is resident in `EA` (typically `f32`) and cast down to the output `E` on
/// drain.
#[cube(launch)]
pub fn cmma_kernel<E: Numeric, EA: Numeric, EL: Numeric, ER: Numeric, D: DeliveryFamily>(
    a: &D::Arg<EL>,
    b: &D::Arg<ER>,
    c: &StridedTileArg<'_, E>,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
    #[define(EA)] _acc_register_dtype: StorageType,
) {
    let a = D::tile::<EL>(a);
    let b = D::tile::<ER>(b);
    let mut c = c.tile();
    let mut acc = c.promote::<EA>();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}
