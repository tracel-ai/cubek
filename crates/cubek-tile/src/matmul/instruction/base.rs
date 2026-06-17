//! The leaf contraction `acc += lhs · rhs`, reached only at a *final* tile: the point where the
//! type-agnostic [lowering](super::lower) stops recursing and the tile finally has to admit it
//! holds numbers. Two peer leaves exist, picked by the accumulator's storage: a cmma fragment runs
//! the hardware instruction ([`cmma`](super::cmma)); plain `Gmem`/`Smem` runs a software
//! microkernel straight over the backing memory ([`register`](super::register)). Both are
//! terminal; neither tiles further.

use cubecl::prelude::*;

use crate::{matmul::instruction::register::mma_register_memory, *};

/// The leaf contraction, keyed on the accumulator's element so the generic lowering can name the
/// bound (`Acc: Mma<Lhs, Rhs>`).
#[cube]
pub trait Mma<Lhs: CubePrimitive, Rhs: CubePrimitive>: CubePrimitive {
    fn mma(acc: &mut Tile<Self>, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>);
}

/// The accumulator's storage picks the backend, and both arms are leaves: a cmma fragment to the
/// tensor-core instruction, plain memory to the register microkernel run in place over the
/// backing `MemData` with no sub-tiling below.
#[cube]
impl<E: Numeric, EL: Numeric, ER: Numeric, V: Size, L: Size> Mma<Vector<EL, L>, Vector<ER, V>>
    for Vector<E, V>
{
    fn mma(acc: &mut Tile<Vector<E, V>>, lhs: &Tile<Vector<EL, L>>, rhs: &Tile<Vector<ER, V>>) {
        let space = comptime!(acc.space.clone());
        let payload = &mut acc.payload;
        match payload {
            Payload::Cmma(d) => d.mma(lhs, rhs),
            Payload::Gmem(g) | Payload::Smem(g) => {
                mma_register_memory::<E, EL, ER, L, V>(g, lhs, rhs, space)
            }
        }
    }
}
