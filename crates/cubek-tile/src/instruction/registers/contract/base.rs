//! The contraction nest's entry point: resolve each operand's quant packing, then route to
//! the 2-D or the N-D nest.

use cubecl::prelude::*;

use super::direct;
use super::gather;
use crate::*;

/// Run the register instruction over each batch matrix, reading operands through the
/// quant-transparent [`matrix_transparent`](Tile::matrix_transparent): a plain operand is a bare
/// matrix read, a quantized one dequantizes per read (no dequantize-into-`f32` fill). Either
/// operand may be the quantized one (the gemv weight is the RHS, an activation-times-weight the
/// LHS), so this dispatches each operand's storage element/packing (`0` plain, `1` native i8, `>1`
/// packed u32). Both quantized at once is not a real workload and is refused.
///
/// The 2-D instruction reads each operand as a batch matrix, which is only a description of it
/// when one axis is contracted *and* a logical coordinate is a physical one. Either condition
/// failing takes the N-D nest ([`gather::contract`](super::gather::contract)); they are independent, so a stencil
/// contracting a single axis is a gather just as much as a two-axis reduce is.
#[cube]
pub(crate) fn memory<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
) {
    let size!(L) = lhs.vector_size();
    let size!(V) = rhs.vector_size();

    // Each operand's storage element is `i8` native / `u32` packed / the served element when plain;
    // its pack factor narrows the physical line, derived in the instruction. `quant_pack` is `0`
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
            gather::contract::<E, EL, i8, L, ER, ER, V>(
                acc, lhs, rhs, space, 1usize, 1usize, config,
            );
        } else if comptime!(pack_l > 1) {
            gather::contract::<E, EL, u32, L, ER, ER, V>(
                acc, lhs, rhs, space, pack_l, 1usize, config,
            );
        } else if comptime!(pack_r == 1) {
            gather::contract::<E, EL, EL, L, ER, i8, V>(
                acc, lhs, rhs, space, 1usize, 1usize, config,
            );
        } else if comptime!(pack_r > 1) {
            gather::contract::<E, EL, EL, L, ER, u32, V>(
                acc, lhs, rhs, space, 1usize, pack_r, config,
            );
        } else {
            gather::contract::<E, EL, EL, L, ER, ER, V>(
                acc, lhs, rhs, space, 1usize, 1usize, config,
            );
        }
    } else if comptime!(pack_l == 1) {
        direct::contract::<E, EL, i8, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize, config);
    } else if comptime!(pack_l > 1) {
        direct::contract::<E, EL, u32, L, ER, ER, V>(acc, lhs, rhs, space, pack_l, 1usize, config);
    } else if comptime!(pack_r == 1) {
        direct::contract::<E, EL, EL, L, ER, i8, V>(acc, lhs, rhs, space, 1usize, 1usize, config);
    } else if comptime!(pack_r > 1) {
        direct::contract::<E, EL, EL, L, ER, u32, V>(acc, lhs, rhs, space, 1usize, pack_r, config);
    } else {
        direct::contract::<E, EL, EL, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize, config);
    }
}
