//! The Cmma kernel: promote the accumulator, contract, copy back.

use cubecl::prelude::*;
use cubek_tile::{BareDelivery, DeliveryFamily, StridedTileArg, Tile, TileSpec};

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
    // The matmul contract is `out = A·B` and `mma` accumulates, so start at zero.
    acc.zero();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}

/// [`cmma_kernel`]'s bare-tensor twin: the identical body over [`BareDelivery`] — plain
/// tensors plus each operand's comptime [`TileSpec`], tiles built in-kernel.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cmma_bare_kernel<
    E: Numeric,
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    VA: Size,
    VB: Size,
    VC: Size,
    D: BareDelivery,
>(
    a: &D::Arg<EL, VA>,
    b: &D::Arg<ER, VB>,
    c: &Tensor<Vector<E, VC>>,
    #[comptime] spec_a: TileSpec,
    #[comptime] spec_b: TileSpec,
    #[comptime] spec_c: TileSpec,
    #[define(EL)] _lhs_dtype: StorageType,
    #[define(ER)] _rhs_dtype: StorageType,
    #[define(E)] _acc_dtype: StorageType,
    #[define(EA)] _acc_register_dtype: StorageType,
) {
    let a = D::tile::<EL, VA>(a, spec_a);
    let b = D::tile::<ER, VB>(b, spec_b);
    let mut c = Tile::<E>::of(c, spec_c);
    let mut acc = c.promote::<EA>();
    // The matmul contract is `out = A·B` and `mma` accumulates, so start at zero.
    acc.zero();
    acc.mma(&a, &b);
    acc.drain_cast_into(&mut c);
}
