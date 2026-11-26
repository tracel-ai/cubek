use super::extents::{ViewDim, ViewRun};
use super::groups::ConvGrouping;
use super::phases::ConvPhasing;
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

    /// The last three of `dimensions` as the one across, the rows and the columns, the rest outer.
    pub fn split(dimensions: &[Dimension]) -> Self {
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

    /// The last two of `dimensions` as the one beside `k` and `k`, the rest outer.
    pub fn split(dimensions: &[Dimension]) -> Self {
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
pub struct SpatialDimensions {
    pub spatial_dims: SpatialDimCount,
    pub grouping: ConvGrouping,
    pub phasing: ConvPhasing,
    pub outs: Vec<Dimension>,
    pub taps: Vec<Dimension>,
}

impl SpatialDimensions {
    pub fn new(
        spatial_dims: SpatialDimCount,
        grouping: ConvGrouping,
        phasing: ConvPhasing,
    ) -> Self {
        let (outs, taps) = ConvDimension::spatial(spatial_dims)
            .map(|[out, tap]| (out, tap))
            .unzip();
        Self {
            spatial_dims,
            grouping,
            phasing,
            outs,
            taps,
        }
    }

    /// The channels dim of `operand`'s view, after the batch or filters, the groups and the
    /// spatial dims.
    pub fn channels(&self, operand: MatmulOperand) -> ViewDim {
        ViewDim::at(
            operand,
            self.spatial_dims.0 + 1 + self.grouping.dimension_count(),
        )
    }

    /// The group dim of the gathered input, the lhs, which is split but never reordered.
    pub fn groups(&self) -> Option<ViewDim> {
        (self.grouping.dimension_count() > 0)
            .then(|| ViewDim::at(MatmulOperand::Lhs, self.spatial_dims.0 + 1))
    }

    /// The spatial dims of `operand`'s view, from `first`.
    pub fn run(&self, operand: MatmulOperand, first: usize) -> ViewRun {
        ViewRun {
            operand,
            first,
            len: self.spatial_dims.0,
        }
    }
}
