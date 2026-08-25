//! The axes and operands every tiled routine builds its space over.

use cubek_tile::{Axis, Operand, OperandSet};

use crate::definition::MatmulElems;

// The matmul tile axes, shared by every tiled routine that lays out its space over `(m, n, k)`
// plus batches. `M`/`N`/`K` are the two matrix dims and the contraction; batch axes follow.
pub(crate) const M: Axis = Axis(0);
pub(crate) const N: Axis = Axis(1);
pub(crate) const K: Axis = Axis(2);

/// The axis for output batch dimension `i` (outermost is `0`).
pub(crate) fn batch_axis(i: usize) -> Axis {
    Axis(3 + i as u8)
}

/// The three matmul operands as [`Operand`]s for a [`Tiling::over`](cubek_tile::Tiling::over)
/// build: `a` over `[M, K]`, `b` over `[K, N]`, `out` over `[M, N]`, each at its global
/// element type. Batch axes are per-binding broadcast facts, stated at launch, not here.
pub(crate) struct MatmulOperands {
    pub a: Operand,
    pub b: Operand,
    pub out: Operand,
}

impl MatmulOperands {
    pub fn new(dtypes: &MatmulElems) -> Self {
        MatmulOperands {
            a: Operand::new(&[M, K], dtypes.lhs_global),
            b: Operand::new(&[K, N], dtypes.rhs_global),
            out: Operand::new(&[M, N], dtypes.acc_global),
        }
    }
}

impl OperandSet for MatmulOperands {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.a, &mut self.b, &mut self.out].into_iter()
    }
}
