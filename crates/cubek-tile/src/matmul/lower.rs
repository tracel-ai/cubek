//! Lowering `c.mma(a, b)`: a tile with levels left lowers per its [`Schedule`]; a final tile
//! contracts via the [`Mma`] leaf.

use cubecl::prelude::*;

use super::schedule::{mma_direct, mma_double, mma_staged};
use crate::matmul::register::mma_register_memory;
use crate::*;

#[cube]
impl<Acc: CubePrimitive> Tile<Acc> {
    /// `c.mma(a, b)`: a tile with levels left lowers per its [`Schedule`]; a final tile
    /// contracts via [`Mma`].
    pub fn mma<Lhs: CubePrimitive, Rhs: CubePrimitive>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>)
    where
        Acc: Mma<Lhs, Rhs>,
    {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final => Acc::mma(self, lhs, rhs),
            Partitioner::Level(level) => {
                // The level's operation space is the merge of the operands' runtime spaces; the
                // output contributes no axis beyond `lhs ∪ rhs`, so the two operands cover it.
                let space = lhs.runtime_space().merge_with(&rhs.runtime_space());
                match level.schedule() {
                    Schedule::Direct => mma_direct(lhs, rhs, self, space),
                    Schedule::Staged => mma_staged(lhs, rhs, self, space),
                    Schedule::DoubleBuffered => mma_double(lhs, rhs, self, space),
                }
            }
        }
    }
}

/// The leaf contraction `acc += lhs · rhs`, reached only at a final tile. Keyed on the
/// accumulator's element so the generic lowering can name the bound; the method takes the whole
/// tile, so it already has the acc's space. The impls that exist are the legal patterns.
#[cube]
pub trait Mma<Lhs: CubePrimitive, Rhs: CubePrimitive>: CubePrimitive {
    fn mma(acc: &mut Tile<Self>, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>);
}

/// Independent operand elements: `lhs` lines `K` by `L`, `rhs`/`acc` line `N` by `V`; each
/// operand is read in its own type and cast to the accumulate element `E` at the leaf, so the
/// same-`E` GEMM is just the `EL = ER = E` case (the casts fold away). `V = L = Const<1>` is the
/// scalar kernel. The accumulator's storage picks the path: cmma fragments via `cmma::execute`,
/// memory tiles via the register microkernel.
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
