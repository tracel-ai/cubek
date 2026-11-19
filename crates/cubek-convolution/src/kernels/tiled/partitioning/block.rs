use cubecl::tile::Space;
use cubek_microkernel::matmul::{MatmulAxes, MatmulStage};

/// What one cube covers of the output, and how deep one stage of the contraction runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvBlock {
    /// Output cells along the rows axis.
    pub rows: usize,
    /// Positions along the across axis, which share each rhs stage.
    pub across: usize,
    /// Output cells along the columns axis.
    pub columns: usize,
    /// Contraction cells one stage holds.
    pub depth: usize,
}

impl ConvBlock {
    /// Output positions the lhs stage holds: every row at every across position.
    pub fn positions(&self) -> usize {
        self.rows * self.across
    }

    /// The box one stage holds, over the rows of every across position (the rhs stage is shared
    /// by them, so only the lhs one grows with them) and the columns.
    pub fn stage(&self, axes: MatmulAxes) -> MatmulStage {
        MatmulStage::new(
            Space::new(&[
                (axes.m, self.positions()),
                (axes.n, self.columns),
                (axes.k, self.depth),
            ]),
            axes,
        )
    }
}
