//! The problem a routine computes.

use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubecl::tune::Work;
use cubek_std::routine::EstimatedProblem;

/// The values of the output, whatever its layout: a draw's answer does not depend on where a value
/// lies, so the space is one axis and every dim of the output projects onto it.
pub const VALUES: Axis = Axis(0);

/// An output filled with draws, as the host describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomRoutineProblem {
    /// The values the output holds.
    pub values: usize,
    /// The element the output is stored in.
    pub element: ElemType,
    /// The lines one draw of the distribution writes.
    pub lines_per_draw: usize,
    /// The alignment of the output's buffer in elements.
    pub alignment: ElemAlignment,
}

impl RandomRoutineProblem {
    /// One axis, as long as the output holds values.
    pub fn space(&self) -> Space {
        Space::new(&[(VALUES, self.values)])
    }

    /// The width the output binds at, at most `widest`: to the requirements the routine
    /// enqueued in `vectorization`, the binding adds the buffer's alignment.
    pub fn width(&self, widest: usize, mut vectorization: Vectorization<()>) -> usize {
        vectorization.require(VectorRequirement::Aligned {
            operand: (),
            alignment: self.alignment,
        });
        vectorization.width((), |_| widest)
    }
}

impl EstimatedProblem for RandomRoutineProblem {
    /// A fill only ever stores: no read, and the arithmetic of a draw is not what bounds it.
    fn demand(&self) -> Work {
        Work {
            compute_ops: 0,
            bytes: self.values * self.element.size(),
        }
    }
}
