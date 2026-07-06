//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`, all three operands fragments.

use cubecl::{cmma, prelude::*};

use crate::*;

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// Tensor-core contraction `self += lhs · rhs` via `cmma::execute`. The operands must be
    /// cmma fragments too.
    pub(crate) fn mma<L: Numeric, R: Numeric>(&self, lhs: &Tile<L>, rhs: &Tile<R>) {
        match (&lhs.tile_kind, &rhs.tile_kind) {
            (TileKind::Cmma(a), TileKind::Cmma(b)) => {
                cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
            }
            _ => panic!("cmma accumulator requires cmma lhs and rhs"),
        }
    }
}
