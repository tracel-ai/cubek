//! The register leaf entry point and shared accumulator helpers.

use cubecl::prelude::*;

use super::direct::mma_register_direct;
use super::gather::mma_register_gather;
use crate::*;

/// Run the register microkernel over each batch matrix, reading operands through the
/// quant-transparent [`matrix_transparent`](Tile::matrix_transparent): a plain operand is a bare
/// matrix read, a quantized one dequantizes per read (no dequantize-into-`f32` fill). Either
/// operand may be the quantized one (the gemv weight is the RHS, an activation-times-weight the
/// LHS), so this dispatches each operand's storage element/packing (`0` plain, `1` native i8, `>1`
/// packed u32). Both quantized at once is not a real workload and is refused.
///
/// The 2-D microkernel reads each operand as a batch matrix, which is only a description of it
/// when one axis is contracted *and* a logical coordinate is a physical one. Either condition
/// failing takes the N-D nest ([`mma_register_gather`]); they are independent, so a stencil
/// contracting a single axis is a gather just as much as a two-axis reduce is.
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] config: MemoryMmaConfig,
) {
    let size!(L) = lhs.vector_size();
    let size!(V) = rhs.vector_size();

    // Each operand's storage element is `i8` native / `u32` packed / the served element when plain;
    // its pack factor narrows the physical line, derived in the microkernel. `quant_pack` is `0`
    // plain / `1` native / `>1` the packed-u32 factor. At most one operand is quantized (the gemv
    // weight is the RHS, an activation·weight the LHS); both at once is refused.
    let pack_l = lhs.quant_pack();
    let pack_r = rhs.quant_pack();
    comptime!(assert!(
        pack_l == 0 || pack_r == 0,
        "register leaf: both operands quantized is not a supported direct-serve case"
    ));

    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let lhs_procedural = lhs.is_procedural();
    let rhs_procedural = rhs.is_procedural();
    let nd = comptime!(
        Space::contracted(&[&lhs.space, &rhs.space], &space).len() > 1
            || lhs_gathered
            || rhs_gathered
            || lhs_procedural
            || rhs_procedural
    );
    if nd {
        if comptime!(pack_l == 1) {
            mma_register_gather::<E, EL, i8, L, ER, ER, V>(
                acc, lhs, rhs, space, 1usize, 1usize, config,
            );
        } else if comptime!(pack_l > 1) {
            mma_register_gather::<E, EL, u32, L, ER, ER, V>(
                acc, lhs, rhs, space, pack_l, 1usize, config,
            );
        } else if comptime!(pack_r == 1) {
            mma_register_gather::<E, EL, EL, L, ER, i8, V>(
                acc, lhs, rhs, space, 1usize, 1usize, config,
            );
        } else if comptime!(pack_r > 1) {
            mma_register_gather::<E, EL, EL, L, ER, u32, V>(
                acc, lhs, rhs, space, 1usize, pack_r, config,
            );
        } else {
            mma_register_gather::<E, EL, EL, L, ER, ER, V>(
                acc, lhs, rhs, space, 1usize, 1usize, config,
            );
        }
    } else if comptime!(pack_l == 1) {
        mma_register_direct::<E, EL, i8, L, ER, ER, V>(
            acc, lhs, rhs, space, 1usize, 1usize, config,
        );
    } else if comptime!(pack_l > 1) {
        mma_register_direct::<E, EL, u32, L, ER, ER, V>(
            acc, lhs, rhs, space, pack_l, 1usize, config,
        );
    } else if comptime!(pack_r == 1) {
        mma_register_direct::<E, EL, EL, L, ER, i8, V>(
            acc, lhs, rhs, space, 1usize, 1usize, config,
        );
    } else if comptime!(pack_r > 1) {
        mma_register_direct::<E, EL, EL, L, ER, u32, V>(
            acc, lhs, rhs, space, 1usize, pack_r, config,
        );
    } else {
        mma_register_direct::<E, EL, EL, L, ER, ER, V>(
            acc, lhs, rhs, space, 1usize, 1usize, config,
        );
    }
}

/// Seed the `mr × nr` register block from the accumulator, once per batch matrix, so the rank-1
/// updates never touch memory. Shared by both microkernels: how a cell is addressed differs, how
/// the block is held does not. `unroll` is the caller's decision, not a size test here, because a
/// masked block must stay rolled whatever its size.
#[cube]
pub(super) fn load_accumulators<E: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            c[i * nr + n] = acc.seed((i as u32, n as u32), comptime!(ReduceLeafKind::Sum));
        }
    }
    c
}

/// The twin of [`load_accumulators`]: commit the block back once the whole reduce is folded into
/// it. Through [`AccumulateView`], so a lane-split accumulator reduces across lanes on the way out
/// rather than the leaf knowing it was split.
#[cube]
pub(super) fn store_accumulators<E: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    c: Array<Vector<E, V>>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) {
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            acc.commit(
                (i as u32, n as u32),
                c[i * nr + n],
                comptime!(ReduceLeafKind::Sum),
            );
        }
    }
}
