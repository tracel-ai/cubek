use super::base::ConvRoles;
use super::groups::ConvGrouping;
use crate::kernels::tiled::SpatialDimCount;
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::matmul::MatmulOperands;

/// The order a view lists the dims of an operand's buffer in, where the roles move the output
/// channels and the groups. The buffer is the tensor with its channels split by group, where the
/// convolution is grouped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewOrder {
    Stored,
    /// The group, stored just before the channels last, listed first.
    GroupsFirst,
    /// The output channels, stored first after the groups, listed last.
    ChannelsOutLast,
    /// The output channels, stored first after the groups, listed just before the input channels.
    ChannelsOutAfterTaps,
}

impl ViewOrder {
    /// The dim of a buffer over `spatial_dims` spatial dims that each dim of its view reads.
    pub fn buffer_dims(self, spatial_dims: SpatialDimCount, grouping: ConvGrouping) -> Vec<usize> {
        let spatial = spatial_dims.0;
        let groups = grouping.dimensions().len();
        let channels = spatial + 1 + groups;
        match self {
            Self::Stored => (0..=channels).collect(),
            Self::GroupsFirst => (spatial + 1..spatial + 1 + groups)
                .chain(0..=spatial)
                .chain([channels])
                .collect(),
            Self::ChannelsOutLast => (0..groups)
                .chain(groups + 1..=channels)
                .chain([groups])
                .collect(),
            Self::ChannelsOutAfterTaps => (0..groups)
                .chain(1 + groups..=spatial + groups)
                .chain([groups, channels])
                .collect(),
        }
    }
}

impl ViewOrder {
    /// `values`, one per dim of a buffer, in the order of its view.
    pub(in crate::kernels::tiled) fn view(
        self,
        spatial_dims: SpatialDimCount,
        grouping: ConvGrouping,
        values: &[usize],
    ) -> Vec<usize> {
        let buffer_dims = self.buffer_dims(spatial_dims, grouping);
        buffer_dims.into_iter().map(|dim| values[dim]).collect()
    }
}

impl ConvRoles {
    /// `shapes`, the operation's operands in their own layouts and split by group where the roles
    /// are grouped, as each is viewed through the roles ([`Self::rhs_view`],
    /// [`Self::out_view`]).
    pub fn views(&self, shapes: &MatmulOperands<Vec<usize>>) -> MatmulOperands<Vec<usize>> {
        let view =
            |order: ViewOrder, shape: &[usize]| order.view(self.spatial_dims, self.grouping, shape);
        MatmulOperands {
            lhs: shapes.lhs.clone(),
            rhs: view(self.rhs_order, &shapes.rhs),
            out: view(self.out_order, &shapes.out),
        }
    }

    /// The alignments of dense buffers of `shapes`, which are already split by group where the
    /// roles are grouped, as the rhs and the output are viewed through the roles
    /// ([`Self::rhs_view`], [`Self::out_view`]).
    pub fn dense_alignments(
        &self,
        shapes: &MatmulOperands<Vec<usize>>,
    ) -> MatmulOperands<ElemAlignment> {
        let dense = |shape: &[usize], order: ViewOrder| {
            let mut strides = vec![1; shape.len()];
            for dim in (0..shape.len() - 1).rev() {
                strides[dim] = strides[dim + 1] * shape[dim + 1];
            }
            ElemAlignment::new(
                &order.view(self.spatial_dims, self.grouping, shape),
                &order.view(self.spatial_dims, self.grouping, &strides),
            )
        };
        MatmulOperands {
            lhs: dense(&shapes.lhs, ViewOrder::Stored),
            rhs: dense(&shapes.rhs, self.rhs_order),
            out: dense(&shapes.out, self.out_order),
        }
    }
}
