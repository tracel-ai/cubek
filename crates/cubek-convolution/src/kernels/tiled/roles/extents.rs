use crate::kernels::tiled::CO;
use cubecl::tile::Dimension;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};

/// One dim of one operand's view, where a problem reads an extent from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewDim {
    pub(super) operand: MatmulOperand,
    pub(super) dim: usize,
}

impl ViewDim {
    /// The extent of this dim in `views`.
    pub fn read(self, views: &MatmulOperands<Vec<usize>>) -> usize {
        self.read_shape(views.get_ref(self.operand))
    }

    /// The extent of this dim in `shape`, the shape of its operand's view.
    pub fn read_shape(self, shape: &[usize]) -> usize {
        shape[self.dim]
    }
}

impl ViewDim {
    pub(super) const fn at(operand: MatmulOperand, dim: usize) -> Self {
        Self { operand, dim }
    }
}

/// The run of dims of one operand's view that hold the spatial extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewRun {
    pub(super) operand: MatmulOperand,
    pub(super) first: usize,
    pub(super) len: usize,
}

impl ViewRun {
    /// The extents of the run in `views`, one per spatial dim.
    pub fn read(self, views: &MatmulOperands<Vec<usize>>) -> &[usize] {
        &views.get_ref(self.operand)[self.first..self.first + self.len]
    }
}

/// Where each extent of a problem lives in the operands' views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvExtents {
    pub batch: ViewDim,
    /// The group dim of the gathered input, where the channels are split by group.
    pub groups: Option<ViewDim>,
    /// The spatial extents of the gathered input.
    pub gathered: ViewRun,
    pub channels_in: ViewDim,
    pub channels_out: ViewDim,
    pub kernel: ViewRun,
    pub out: ViewRun,
}

impl ConvExtents {
    /// The channels of the gathered input, whichever of the input and output channels `inner` names.
    pub fn lhs_channels(&self, inner: Dimension) -> ViewDim {
        match inner == CO {
            true => self.channels_out,
            false => self.channels_in,
        }
    }
}
