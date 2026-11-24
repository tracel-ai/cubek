use super::dimensions::{ConvContractedDimensions, ConvFreeDimensions, SpatialDimensions};
use super::extents::{ConvExtents, ViewDim};
use super::gather::ConvGather;
use super::groups::{ConvChannelOperands, ConvGrouping};
use super::views::ViewOrder;
use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{B, CI, CO, ConvDecline, ConvDimension, SpatialDimCount};
use cubecl::prelude::TensorBinding;
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubek_microkernel::matmul::{MatmulDimensions, MatmulLayout, MatmulOperand, MatmulOperands};

/// Which dimension of the space plays which part in an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvRoles {
    /// The spatial dims of the convolution.
    pub spatial_dims: SpatialDimCount,
    /// Whether the channels are split by group: the group is then the first free dimension.
    pub grouping: ConvGrouping,
    /// Where the group count is read from.
    pub channels: ConvChannelOperands,
    pub free: ConvFreeDimensions,
    pub contracted: ConvContractedDimensions,
    /// The dimensions of the space, the rows before the columns: the planes of a cube take them in
    /// the space's order.
    pub space: Vec<Dimension>,
    /// The dimensions the gathered input is read over, the contracted run just before the free one:
    /// a matrix instruction takes its operands' dimensions in that order.
    pub lhs: Vec<Dimension>,
    /// The dimensions the rhs is bound by, in the order the instruction reads them.
    pub rhs: Vec<Dimension>,
    /// The order the rhs's view lists the dims of its buffer in, as split by group.
    pub rhs_order: ViewOrder,
    /// The order the output's view lists the dims of its buffer in.
    pub out_order: ViewOrder,
    /// The dimension each operand's buffer is contiguous along: the one its vectors run along.
    pub inner: MatmulOperands<Dimension>,
    /// How the gathered input is addressed.
    pub gather: ConvGather,
    pub extents: ConvExtents,
    /// What the window's extents are declined with where they do not yield the output's.
    pub extent_mismatch: ConvDecline,
}

/// The dimensions of `parts` one after the other; the slices let arrays and `Vec`s mix in one list.
fn joined(parts: &[&[Dimension]]) -> Vec<Dimension> {
    parts.concat()
}

impl ConvRoles {
    /// The roles `operation` gives the dimensions of the space of a convolution over `spatial_dims`
    /// spatial dims, with the channels as `grouping` says.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::NoSpatialDim`], for no spatial dim, [`ConvDecline::TooManySpatialDims`], for
    /// more than [`ConvDimension::MAX_SPATIAL_DIMS`], and [`ConvDecline::Transposed`], for a
    /// transposed forward.
    pub fn of(
        operation: ConvolutionOperation,
        spatial_dims: SpatialDimCount,
        grouping: ConvGrouping,
    ) -> Result<Self, Decline> {
        if spatial_dims.0 == 0 {
            return Err(ConvDecline::NoSpatialDim.into());
        }
        if spatial_dims.0 > ConvDimension::MAX_SPATIAL_DIMS {
            return Err(ConvDecline::TooManySpatialDims.into());
        }
        let channels = ConvChannelOperands::of(operation)?;
        let spatial = SpatialDimensions::new(spatial_dims, grouping);
        let build = match operation {
            ConvolutionOperation::Forward => Self::forward,
            ConvolutionOperation::BackwardWeight => Self::backward_weight,
            ConvolutionOperation::BackwardData | ConvolutionOperation::ForwardTransposed => {
                Self::backward_data
            }
        };
        Ok(build(&spatial, channels))
    }

    /// The dimensions a forward's bias is bound by: one value for each output channel of each
    /// group.
    pub fn bias_dimensions(&self) -> Vec<Dimension> {
        joined(&[self.grouping.dimensions(), &[CO]])
    }

    /// The spatial `[out, tap]` dimension pairs, one per spatial dim of the gathered input.
    pub fn spatial(&self) -> impl Iterator<Item = [Dimension; 2]> {
        ConvDimension::spatial(self.spatial_dims)
    }

    /// The dimension a kernel leaves to its launch: the batch where it is a free dimension. The
    /// weight backward contracts it, so it stays compiled in there.
    pub fn dynamic(&self) -> Option<Dimension> {
        self.free.in_order().contains(&B).then_some(B)
    }

    /// How each operand's buffer lies against the product's own `[m, k]`, `[k, n]` and `[m, n]`:
    /// row where it runs along the second of them, col along the first.
    pub fn layouts(&self) -> MatmulOperands<MatmulLayout> {
        let dimensions = self.dimensions();
        let layout = |operand: MatmulOperand| {
            let along = dimensions.dim(self.inner.get(operand));
            MatmulLayout::along(along.map(|dim| MatmulDimensions::MNK.of(dim)), operand)
        };
        MatmulOperands {
            lhs: layout(MatmulOperand::Lhs),
            rhs: layout(MatmulOperand::Rhs),
            out: layout(MatmulOperand::Out),
        }
    }

    /// The dimensions of the instruction's rows, columns and contracted cells.
    pub fn dimensions(&self) -> MatmulDimensions {
        MatmulDimensions {
            m: self.free.m,
            n: self.free.n,
            k: self.contracted.k,
        }
    }

    /// Whether `rhs` and `out`, viewed through the roles, and the input's `extent` of `dimension`
    /// (`None` where the input does not carry it) all give `dimension` the same extent.
    pub fn agree_on(
        &self,
        dimension: Dimension,
        extent: Option<usize>,
        [rhs, out]: [&TensorBinding; 2],
    ) -> bool {
        let free = self.free.in_order();
        let named = [(&self.rhs[..], rhs), (&free[..], out)];
        let mut extents =
            extent
                .into_iter()
                .chain(named.iter().filter_map(|(dimensions, tensor)| {
                    let dim = dimensions.iter().position(|&named| named == dimension)?;
                    Some(tensor.shape[dim])
                }));
        let first = extents.next();
        extents.all(|extent| Some(extent) == first)
    }

    /// The dimensions a plane window launch lays its planes over: the rows, the columns, the one
    /// across and the outer ones.
    pub fn plane_dimensions(&self) -> Vec<Dimension> {
        let free = &self.free;
        joined(&[&[free.m, free.n, free.across], &free.outer[..]])
    }

    /// `rhs` with its dims listed in the order of [`Self::rhs`].
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the dims move and `rhs` is tiled.
    pub fn rhs_view(&self, rhs: TensorBinding) -> Result<TensorBinding, Decline> {
        self.rhs_order
            .permute(rhs, self.spatial_dims, self.grouping)
    }

    /// `out`, a tensor of the operation's output in its own layout, with its dims listed in the
    /// order of [`Self::free`].
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the dims move and `out` is tiled.
    pub fn out_view(&self, out: TensorBinding) -> Result<TensorBinding, Decline> {
        self.out_order
            .permute(out, self.spatial_dims, self.grouping)
    }

    /// `tensor`, the `operand` of the operation in its own layout, with its channels split by
    /// group where the roles are grouped.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the channels split and `tensor` is tiled.
    pub fn split(
        &self,
        operand: MatmulOperand,
        tensor: TensorBinding,
    ) -> Result<TensorBinding, Decline> {
        self.channels.axis(operand).split(tensor, self.grouping)
    }
}

impl ConvRoles {
    fn forward(spatial: &SpatialDimensions, channels: ConvChannelOperands) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let groups = spatial.grouping.dimensions();
        let split = spatial.split_dims();
        let free = joined(&[groups, &[B], &spatial.outs, &[CO]]);
        let contracted = joined(&[&spatial.taps, &[CI]]);
        Self {
            spatial_dims: spatial.spatial_dims,
            grouping: spatial.grouping,
            channels,
            space: joined(&[&free, &contracted]),
            free: ConvFreeDimensions::split(&free),
            contracted: ConvContractedDimensions::split(&contracted),
            lhs: joined(&[groups, &[B], &spatial.outs, &spatial.taps, &[CI]]),
            rhs: joined(&[groups, &[CO], &spatial.taps, &[CI]]),
            rhs_order: ViewOrder::Stored,
            out_order: ViewOrder::GroupsFirst,
            inner: MatmulOperands {
                lhs: CI,
                rhs: CI,
                out: CO,
            },
            gather: ConvGather::Window,
            extents: ConvExtents {
                batch: ViewDim::at(Lhs, 0),
                groups: spatial.groups(),
                lhs_channels: spatial.channels(Lhs),
                gathered: spatial.run(Lhs, 1),
                channels_in: spatial.channels(Lhs),
                channels_out: ViewDim::at(Rhs, split),
                kernel: spatial.run(Rhs, 1 + split),
                out: spatial.run(Out, 1 + split),
            },
            extent_mismatch: ConvDecline::ForwardExtent,
        }
    }

    fn backward_weight(spatial: &SpatialDimensions, channels: ConvChannelOperands) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let groups = spatial.grouping.dimensions();
        let split = spatial.split_dims();
        let free = joined(&[groups, &spatial.taps, &[CI, CO]]);
        let contracted = joined(&[&[B], &spatial.outs]);
        Self {
            spatial_dims: spatial.spatial_dims,
            grouping: spatial.grouping,
            channels,
            space: joined(&[&contracted, &free]),
            free: ConvFreeDimensions::split(&free),
            contracted: ConvContractedDimensions::split(&contracted),
            lhs: joined(&[groups, &spatial.taps, &[B], &spatial.outs, &[CI]]),
            rhs: joined(&[groups, &[B], &spatial.outs, &[CO]]),
            rhs_order: ViewOrder::GroupsFirst,
            out_order: ViewOrder::ChannelsOutLast,
            inner: MatmulOperands {
                lhs: CI,
                rhs: CO,
                out: CI,
            },
            gather: ConvGather::Window,
            extents: ConvExtents {
                batch: ViewDim::at(Lhs, 0),
                groups: spatial.groups(),
                lhs_channels: spatial.channels(Lhs),
                gathered: spatial.run(Lhs, 1),
                channels_in: spatial.channels(Lhs),
                channels_out: spatial.channels(Out),
                kernel: spatial.run(Out, split),
                out: spatial.run(Rhs, 1 + split),
            },
            extent_mismatch: ConvDecline::WeightGradExtent,
        }
    }

    fn backward_data(spatial: &SpatialDimensions, channels: ConvChannelOperands) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let groups = spatial.grouping.dimensions();
        let split = spatial.split_dims();
        let free = joined(&[groups, &[B], &spatial.outs, &[CI]]);
        let contracted = joined(&[&spatial.taps, &[CO]]);
        Self {
            spatial_dims: spatial.spatial_dims,
            grouping: spatial.grouping,
            channels,
            space: joined(&[&free, &contracted]),
            free: ConvFreeDimensions::split(&free),
            contracted: ConvContractedDimensions::split(&contracted),
            lhs: joined(&[groups, &[B], &spatial.outs, &spatial.taps, &[CO]]),
            rhs: joined(&[groups, &spatial.taps, &[CO, CI]]),
            rhs_order: ViewOrder::ChannelsOutAfterTaps,
            out_order: ViewOrder::GroupsFirst,
            inner: MatmulOperands {
                lhs: CO,
                rhs: CI,
                out: CI,
            },
            gather: ConvGather::Gradient,
            extents: ConvExtents {
                batch: ViewDim::at(Lhs, 0),
                groups: spatial.groups(),
                lhs_channels: spatial.channels(Lhs),
                gathered: spatial.run(Lhs, 1),
                channels_in: spatial.channels(Out),
                channels_out: spatial.channels(Lhs),
                kernel: spatial.run(Rhs, split),
                out: spatial.run(Out, 1 + split),
            },
            extent_mismatch: ConvDecline::DataGradExtent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tiled::G;
    use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};
    use std::collections::HashSet;

    const OPERATIONS: [ConvolutionOperation; 3] = [Forward, BackwardWeight, BackwardData];

    /// A transposed forward is no operation of the tiled convolution, which states it as the data
    /// backward at a stride.
    #[test]
    fn the_transposed_forward_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(
                ConvolutionOperation::ForwardTransposed,
                SpatialDimCount(2),
                ConvGrouping::Whole
            ),
            Err(ConvDecline::Transposed.into())
        );
    }

    #[test]
    fn no_spatial_dim_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(Forward, SpatialDimCount(0), ConvGrouping::Whole),
            Err(ConvDecline::NoSpatialDim.into())
        );
    }

    #[test]
    fn spatial_dims_past_the_label_range_are_refused_by_name() {
        let max = ConvDimension::MAX_SPATIAL_DIMS;
        assert!(ConvRoles::of(Forward, SpatialDimCount(max), ConvGrouping::Whole).is_ok());
        assert_eq!(
            ConvRoles::of(Forward, SpatialDimCount(max + 1), ConvGrouping::Whole),
            Err(ConvDecline::TooManySpatialDims.into())
        );
    }

    /// Over two spatial dims the roles are the 2D tables.
    #[test]
    fn two_spatial_dims_state_the_two_dimensional_tables() {
        let [[oh, rh], [ow, rw]] = ConvDimension::spatial(SpatialDimCount(2))
            .collect::<Vec<_>>()
            .try_into()
            .expect("two spatial dims");
        let roles = |operation| {
            ConvRoles::of(operation, SpatialDimCount(2), ConvGrouping::Whole)
                .expect("a served operation")
        };
        let whole = ConvGrouping::Whole;

        let forward = roles(Forward);
        assert_eq!(forward.free.in_order(), [B, oh, ow, CO]);
        assert_eq!(forward.contracted.in_order(), [rh, rw, CI]);
        assert_eq!(forward.space, [B, oh, ow, CO, rh, rw, CI]);
        assert_eq!(forward.lhs, [B, oh, ow, rh, rw, CI]);
        assert_eq!(forward.rhs, [CO, rh, rw, CI]);
        assert_eq!(
            forward.rhs_order.buffer_dims(SpatialDimCount(2), whole),
            [0, 1, 2, 3]
        );
        assert_eq!(
            forward.out_order.buffer_dims(SpatialDimCount(2), whole),
            [0, 1, 2, 3]
        );
        assert_eq!(
            forward.inner,
            MatmulOperands {
                lhs: CI,
                rhs: CI,
                out: CO
            }
        );
        assert_eq!(forward.gather, ConvGather::Window);
        assert_eq!(forward.extent_mismatch, ConvDecline::ForwardExtent);

        let weight = roles(BackwardWeight);
        assert_eq!(weight.free.in_order(), [rh, rw, CI, CO]);
        assert_eq!(weight.contracted.in_order(), [B, oh, ow]);
        assert_eq!(weight.space, [B, oh, ow, rh, rw, CI, CO]);
        assert_eq!(weight.lhs, [rh, rw, B, oh, ow, CI]);
        assert_eq!(weight.rhs, [B, oh, ow, CO]);
        assert_eq!(
            weight.rhs_order.buffer_dims(SpatialDimCount(2), whole),
            [0, 1, 2, 3]
        );
        assert_eq!(
            weight.out_order.buffer_dims(SpatialDimCount(2), whole),
            [1, 2, 3, 0]
        );
        assert_eq!(
            weight.inner,
            MatmulOperands {
                lhs: CI,
                rhs: CO,
                out: CI
            }
        );
        assert_eq!(weight.gather, ConvGather::Window);
        assert_eq!(weight.extent_mismatch, ConvDecline::WeightGradExtent);

        let data = roles(BackwardData);
        assert_eq!(data.free.in_order(), [B, oh, ow, CI]);
        assert_eq!(data.contracted.in_order(), [rh, rw, CO]);
        assert_eq!(data.space, [B, oh, ow, CI, rh, rw, CO]);
        assert_eq!(data.lhs, [B, oh, ow, rh, rw, CO]);
        assert_eq!(data.rhs, [rh, rw, CO, CI]);
        assert_eq!(
            data.rhs_order.buffer_dims(SpatialDimCount(2), whole),
            [1, 2, 0, 3]
        );
        assert_eq!(
            data.out_order.buffer_dims(SpatialDimCount(2), whole),
            [0, 1, 2, 3]
        );
        assert_eq!(
            data.inner,
            MatmulOperands {
                lhs: CO,
                rhs: CI,
                out: CI
            }
        );
        assert_eq!(data.gather, ConvGather::Gradient);
        assert_eq!(data.extent_mismatch, ConvDecline::DataGradExtent);
    }

    /// Over two spatial dims, grouped, the group leads the free dimensions of every operation and
    /// the views list it first.
    #[test]
    fn grouped_roles_put_the_group_first_among_the_free_dimensions() {
        let grouped = ConvGrouping::Grouped(2);
        let [[oh, rh], [ow, rw]] = ConvDimension::spatial(SpatialDimCount(2))
            .collect::<Vec<_>>()
            .try_into()
            .expect("two spatial dims");
        let roles = |operation| {
            ConvRoles::of(operation, SpatialDimCount(2), grouped).expect("a served operation")
        };
        let dims = SpatialDimCount(2);

        let forward = roles(Forward);
        assert_eq!(forward.free.in_order(), [G, B, oh, ow, CO]);
        assert_eq!(forward.free.outer, [G, B]);
        assert_eq!(forward.lhs, [G, B, oh, ow, rh, rw, CI]);
        assert_eq!(forward.rhs, [G, CO, rh, rw, CI]);
        assert_eq!(forward.bias_dimensions(), [G, CO]);
        assert_eq!(
            forward.rhs_order.buffer_dims(dims, grouped),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            forward.out_order.buffer_dims(dims, grouped),
            [3, 0, 1, 2, 4]
        );

        let weight = roles(BackwardWeight);
        assert_eq!(weight.free.in_order(), [G, rh, rw, CI, CO]);
        assert_eq!(weight.contracted.in_order(), [B, oh, ow]);
        assert_eq!(weight.rhs, [G, B, oh, ow, CO]);
        assert_eq!(weight.rhs_order.buffer_dims(dims, grouped), [3, 0, 1, 2, 4]);
        assert_eq!(weight.out_order.buffer_dims(dims, grouped), [0, 2, 3, 4, 1]);

        let data = roles(BackwardData);
        assert_eq!(data.free.in_order(), [G, B, oh, ow, CI]);
        assert_eq!(data.rhs, [G, rh, rw, CO, CI]);
        assert_eq!(data.rhs_order.buffer_dims(dims, grouped), [0, 2, 3, 1, 4]);
        assert_eq!(data.out_order.buffer_dims(dims, grouped), [3, 0, 1, 2, 4]);
    }

    /// Whatever the spatial count and the grouping, every dimension of the space is named once, the
    /// operands' dimensions cover it, and a view lists each dim of its buffer once.
    #[test]
    fn roles_of_any_spatial_count_name_each_dimension_once() {
        for operation in OPERATIONS {
            for count in [1, 3] {
                for grouping in [ConvGrouping::Whole, ConvGrouping::Grouped(2)] {
                    let spatial_dims = SpatialDimCount(count);
                    let roles = ConvRoles::of(operation, spatial_dims, grouping)
                        .expect("a served operation");
                    let groups = grouping.dimensions().len();
                    let space: HashSet<Dimension> = roles.space.iter().copied().collect();
                    assert_eq!(space.len(), roles.space.len(), "{operation:?} {count}");
                    assert_eq!(space.len(), 2 * count + 3 + groups, "{operation:?} {count}");
                    let named: HashSet<Dimension> =
                        [&roles.lhs, &roles.rhs, &roles.free.in_order()]
                            .into_iter()
                            .flatten()
                            .copied()
                            .collect();
                    assert_eq!(named, space, "{operation:?} {count}");
                    for order in [roles.rhs_order, roles.out_order] {
                        let mut listed = order.buffer_dims(spatial_dims, grouping);
                        listed.sort_unstable();
                        assert_eq!(listed, (0..count + 2 + groups).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}
