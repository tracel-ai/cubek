use super::groups::ConvGrouping;
use crate::kernels::tiled::SpatialDimCount;

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
        let groups = grouping.dimension_count();
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
    pub fn view(
        self,
        spatial_dims: SpatialDimCount,
        grouping: ConvGrouping,
        values: &[usize],
    ) -> Vec<usize> {
        let buffer_dims = self.buffer_dims(spatial_dims, grouping);
        buffer_dims.into_iter().map(|dim| values[dim]).collect()
    }
}
