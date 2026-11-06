use super::{B, ConvOperation, OH, OW, RH, RW};
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::tile::layout::Scale;
use cubecl::tile::{Axis, Projection};

/// Where each output position reads its input: `position = out * stride + tap * dilation -
/// padding` along each spatial dimension. The data backward reads the gradient the other way,
/// `position = out - tap * dilation + padding`, at stride 1. The one home of those formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvWindow {
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    /// Padding before each spatial dimension.
    pub padding: [usize; 2],
}

impl ConvWindow {
    /// Whether a position can read before the input's start.
    pub fn underflows(&self, operation: ConvOperation) -> bool {
        match operation {
            ConvOperation::BackwardData => true,
            ConvOperation::Forward | ConvOperation::BackwardWeight => {
                self.padding.iter().any(|&padding| padding > 0)
            }
        }
    }

    /// The input as the contraction reads it, over the six axes `axes` lists in the order a
    /// matrix instruction needs them (the contracted run beside the free `ci`): an NHWC buffer
    /// whose two spatial dims are each addressed by an output axis and a tap axis.
    pub fn input(&self, operation: ConvOperation, axes: [Axis; 6]) -> Projection {
        let along = |out, tap, dim: usize| match operation {
            ConvOperation::BackwardData => PhysicalAxisMap::scaled(&[
                (out, Scale::Static(1)),
                (tap, Scale::Static(-(self.dilation[dim] as isize))),
            ])
            .shifted(self.padding[dim] as isize),
            _ => PhysicalAxisMap::affine(&[(out, self.stride[dim]), (tap, self.dilation[dim])])
                .shifted(-(self.padding[dim] as isize)),
        };
        Projection::new(
            &axes,
            &[
                PhysicalAxisMap::of(B),
                along(OH, RH, 0),
                along(OW, RW, 1),
                PhysicalAxisMap::of(operation.roles().inner.lhs),
            ],
        )
    }
}
