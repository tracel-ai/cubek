use super::extents::{ViewDim, ViewRun};
use super::groups::ConvGrouping;
use crate::kernels::tiled::{ConvDimension, SpatialDimCount};
use cubecl::tile::Dimension;
use cubek_microkernel::matmul::MatmulOperand;

/// The output's dimensions by the part each plays. The outer dimensions and the one across take a
/// tile each of a launch's cubes or planes, after the rows and the columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvFreeDimensions {
    pub outer: Vec<Dimension>,
    pub across: Dimension,
    pub m: Dimension,
    pub n: Dimension,
}

impl ConvFreeDimensions {
    /// The dimensions in the order the instruction reads them.
    pub fn in_order(&self) -> Vec<Dimension> {
        [&self.outer[..], &[self.across, self.m, self.n]].concat()
    }
}

impl ConvFreeDimensions {
    /// The last three of `dimensions` as the one across, the rows and the columns, the rest outer.
    pub(super) fn split(dimensions: &[Dimension]) -> Self {
        let [outer @ .., across, m, n] = dimensions else {
            panic!("a convolution's output has an across, a rows and a columns dimension");
        };
        Self {
            outer: outer.to_vec(),
            across: *across,
            m: *m,
            n: *n,
        }
    }
}

/// The contracted dimensions a box walks whole. `beside` fills a step of the contraction where `k`
/// alone is shorter than the instruction's `k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvContractedDimensions {
    pub outer: Vec<Dimension>,
    pub beside: Dimension,
    pub k: Dimension,
}

impl ConvContractedDimensions {
    /// The dimensions in the order a box walks them.
    pub fn in_order(&self) -> Vec<Dimension> {
        [&self.outer[..], &[self.beside, self.k]].concat()
    }
}

impl ConvContractedDimensions {
    /// The last two of `dimensions` as the one beside `k` and `k`, the rest outer.
    pub(super) fn split(dimensions: &[Dimension]) -> Self {
        let [outer @ .., beside, k] = dimensions else {
            panic!("a convolution contracts a beside and a k dimension");
        };
        Self {
            outer: outer.to_vec(),
            beside: *beside,
            k: *k,
        }
    }
}

/// The out and tap dimensions of a convolution over `spatial_dims` spatial dims, and where their
/// extents sit in the operands' views.
#[derive(Debug)]
pub(super) struct SpatialDimensions {
    pub(super) spatial_dims: SpatialDimCount,
    pub(super) grouping: ConvGrouping,
    pub(super) outs: Vec<Dimension>,
    pub(super) taps: Vec<Dimension>,
}

impl SpatialDimensions {
    pub(super) fn new(spatial_dims: SpatialDimCount, grouping: ConvGrouping) -> Self {
        let (outs, taps) = ConvDimension::spatial(spatial_dims)
            .map(|[out, tap]| (out, tap))
            .unzip();
        Self {
            spatial_dims,
            grouping,
            outs,
            taps,
        }
    }

    /// The dims the split by group adds to a view: none, or the group.
    pub(super) fn split_dims(&self) -> usize {
        self.grouping.dimensions().len()
    }

    /// The channels dim of `operand`'s view, after the batch or filters, the groups and the
    /// spatial dims.
    pub(super) fn channels(&self, operand: MatmulOperand) -> ViewDim {
        ViewDim::at(operand, self.spatial_dims.0 + 1 + self.split_dims())
    }

    /// The group dim of the gathered input, the lhs, which is split but never reordered.
    pub(super) fn groups(&self) -> Option<ViewDim> {
        (self.split_dims() > 0).then(|| ViewDim::at(MatmulOperand::Lhs, self.spatial_dims.0 + 1))
    }

    /// The spatial dims of `operand`'s view, from `first`.
    pub(super) fn run(&self, operand: MatmulOperand, first: usize) -> ViewRun {
        ViewRun {
            operand,
            first,
            len: self.spatial_dims.0,
        }
    }
}
