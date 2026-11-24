use super::dimensions::{ConvContractedDimensions, ConvFreeDimensions, SpatialDimensions};
use super::extents::{ConvExtents, ViewDim};
use super::gather::ConvGather;
use super::groups::{ConvChannelOperands, ConvGrouping};
use super::phases::ConvPhasing;
use super::views::ViewOrder;
use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{B, CI, CO, ConvDecline, ConvDimension, SpatialDimCount};
use crate::launch::SpatialWindow;
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::matmul::{MatmulDimensions, MatmulLayout, MatmulOperand, MatmulOperands};

/// Which dimension of the space plays which part in an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvRoles {
    /// The operation these roles are of.
    pub operation: ConvolutionOperation,
    /// The spatial dims of the convolution.
    pub spatial_dims: SpatialDimCount,
    /// Whether the channels are split by group: the group is then the first free dimension.
    pub grouping: ConvGrouping,
    /// Whether the strided dims are split into phases: each phase is then a free dimension, after
    /// the group, that the rhs and the output are bound over and the gathered input is not.
    pub phasing: ConvPhasing,
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
    /// The dimensions the rhs is bound by, in the order the instruction reads them. A phase
    /// dimension is not an axis of its buffer: a tap axis is addressed by a phase and a tap.
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

impl ConvRoles {
    /// The roles `operation` gives the dimensions of the space of a convolution with one spatial
    /// dim for each of `windows`, with the channels as `grouping` says. Only a data backward over
    /// a strided window is split into phases ([`ConvPhasing::of`]).
    ///
    /// # Errors
    ///
    /// [`ConvDecline::NoSpatialDim`], for no spatial dim, [`ConvDecline::TooManySpatialDims`], for
    /// more than [`ConvDimension::MAX_SPATIAL_DIMS`], and [`ConvDecline::Transposed`], for a
    /// transposed forward.
    pub fn of(
        operation: ConvolutionOperation,
        grouping: ConvGrouping,
        windows: &[SpatialWindow],
    ) -> Result<Self, Decline> {
        let spatial_dims = SpatialDimCount(windows.len());
        if spatial_dims.0 == 0 {
            return Err(ConvDecline::NoSpatialDim.into());
        }
        if spatial_dims.0 > ConvDimension::MAX_SPATIAL_DIMS {
            return Err(ConvDecline::TooManySpatialDims.into());
        }
        let channels = ConvChannelOperands::of(operation)?;
        let phasing = ConvPhasing::of(operation, windows.iter().copied());
        let spatial = SpatialDimensions::new(spatial_dims, grouping, phasing);
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
        [self.grouping.dimensions(), &[CO]].concat()
    }

    /// The dimensions of the rhs buffer's own axes: [`Self::rhs`] without the phases.
    pub fn rhs_axes(&self) -> Vec<Dimension> {
        self.without_phases(&self.rhs)
    }

    /// The dimensions of the output buffer's own axes: [`Self::free`] without the phases.
    pub fn out_axes(&self) -> Vec<Dimension> {
        self.without_phases(&self.free.in_order())
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

    /// The dimensions a plane window launch lays its planes over: the rows, the columns, the one
    /// across and the outer ones.
    pub fn plane_dimensions(&self) -> Vec<Dimension> {
        let free = &self.free;
        [&[free.m, free.n, free.across], &free.outer[..]].concat()
    }

    /// `shapes`, the operation's operands in their own layouts and split by group where the roles
    /// are grouped, as each is viewed through the roles' `rhs_order` and
    /// `out_order`.
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
    /// roles are grouped, as the rhs and the output are viewed through the roles'
    /// `rhs_order` and `out_order`.
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

impl ConvRoles {
    /// `dimensions` without the phase dimensions.
    fn without_phases(&self, dimensions: &[Dimension]) -> Vec<Dimension> {
        let phases = self.phasing.dimensions();
        (dimensions.iter().copied())
            .filter(|dimension| !phases.contains(dimension))
            .collect()
    }

    fn forward(spatial: &SpatialDimensions, channels: ConvChannelOperands) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let groups = spatial.grouping.dimensions();
        let split = spatial.grouping.dimension_count();
        let free = [groups, &[B], &spatial.outs, &[CO]].concat();
        let contracted = [&spatial.taps[..], &[CI]].concat();
        Self {
            operation: ConvolutionOperation::Forward,
            spatial_dims: spatial.spatial_dims,
            grouping: spatial.grouping,
            phasing: ConvPhasing::Whole,
            channels,
            space: [&free[..], &contracted].concat(),
            free: ConvFreeDimensions::split(&free),
            contracted: ConvContractedDimensions::split(&contracted),
            lhs: [groups, &[B], &spatial.outs, &spatial.taps, &[CI]].concat(),
            rhs: [groups, &[CO], &spatial.taps, &[CI]].concat(),
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
        let split = spatial.grouping.dimension_count();
        let free = [groups, &spatial.taps, &[CI, CO]].concat();
        let contracted = [&[B][..], &spatial.outs].concat();
        Self {
            operation: ConvolutionOperation::BackwardWeight,
            spatial_dims: spatial.spatial_dims,
            grouping: spatial.grouping,
            phasing: ConvPhasing::Whole,
            channels,
            space: [&contracted[..], &free].concat(),
            free: ConvFreeDimensions::split(&free),
            contracted: ConvContractedDimensions::split(&contracted),
            lhs: [groups, &spatial.taps, &[B], &spatial.outs, &[CI]].concat(),
            rhs: [groups, &[B], &spatial.outs, &[CO]].concat(),
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
        let split = spatial.grouping.dimension_count();
        let free = [groups, &spatial.phases, &[B], &spatial.outs, &[CI]].concat();
        let contracted = [&spatial.taps[..], &[CO]].concat();
        Self {
            operation: ConvolutionOperation::BackwardData,
            spatial_dims: spatial.spatial_dims,
            grouping: spatial.grouping,
            phasing: spatial.phasing.clone(),
            channels,
            space: [&free[..], &contracted].concat(),
            free: ConvFreeDimensions::split(&free),
            contracted: ConvContractedDimensions::split(&contracted),
            lhs: [groups, &[B], &spatial.outs, &spatial.taps, &[CO]].concat(),
            rhs: [groups, &spatial.phases, &spatial.taps, &[CO, CI]].concat(),
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

    /// `strides.len()` windows of those strides.
    fn windows(strides: &[usize]) -> Vec<SpatialWindow> {
        (strides.iter())
            .map(|&stride| SpatialWindow {
                stride,
                padding: 1,
                dilation: 1,
            })
            .collect()
    }

    /// `count` spatial dims at stride 1.
    fn unit(count: usize) -> Vec<SpatialWindow> {
        windows(&vec![1; count])
    }

    /// A transposed forward is no operation of the tiled convolution, which states it as the data
    /// backward at a stride.
    #[test]
    fn the_transposed_forward_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(
                ConvolutionOperation::ForwardTransposed,
                ConvGrouping::Whole,
                &unit(2)
            ),
            Err(ConvDecline::Transposed.into())
        );
    }

    #[test]
    fn no_spatial_dim_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(Forward, ConvGrouping::Whole, &unit(0)),
            Err(ConvDecline::NoSpatialDim.into())
        );
    }

    #[test]
    fn spatial_dims_past_the_label_range_are_refused_by_name() {
        let max = ConvDimension::MAX_SPATIAL_DIMS;
        assert!(ConvRoles::of(Forward, ConvGrouping::Whole, &unit(max)).is_ok());
        assert_eq!(
            ConvRoles::of(Forward, ConvGrouping::Whole, &unit(max + 1)),
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
            ConvRoles::of(operation, ConvGrouping::Whole, &unit(2)).expect("a served operation")
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
        let roles =
            |operation| ConvRoles::of(operation, grouped, &unit(2)).expect("a served operation");
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
                    let roles = ConvRoles::of(operation, grouping, &unit(count))
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

    /// A strided data backward adds a phase dimension to the free and rhs dimensions of each
    /// strided dim, after the group, and to nothing else.
    #[test]
    fn a_strided_data_backward_adds_a_free_phase_dimension_for_each_strided_dim() {
        let [[oh, rh], [ow, rw]] = ConvDimension::spatial(SpatialDimCount(2))
            .collect::<Vec<_>>()
            .try_into()
            .expect("two spatial dims");
        let (ph, pw) = (
            ConvDimension::Phase(0).dimension(),
            ConvDimension::Phase(1).dimension(),
        );
        let roles = |grouping, strides: Vec<usize>| {
            ConvRoles::of(BackwardData, grouping, &windows(&strides)).expect("a served operation")
        };

        let both = roles(ConvGrouping::Whole, vec![2, 2]);
        assert_eq!(both.free.in_order(), [ph, pw, B, oh, ow, CI]);
        assert_eq!(both.free.outer, [ph, pw, B]);
        assert_eq!(both.out_axes(), [B, oh, ow, CI]);
        assert_eq!(both.rhs, [ph, pw, rh, rw, CO, CI]);
        assert_eq!(both.rhs_axes(), [rh, rw, CO, CI]);
        assert_eq!(both.lhs, [B, oh, ow, rh, rw, CO]);
        assert_eq!(both.space, [ph, pw, B, oh, ow, CI, rh, rw, CO]);
        assert_eq!(both.dynamic(), Some(B));

        let one = roles(ConvGrouping::Whole, vec![1, 3]);
        assert_eq!(one.free.in_order(), [pw, B, oh, ow, CI]);
        assert_eq!(one.rhs, [pw, rh, rw, CO, CI]);

        let grouped = roles(ConvGrouping::Grouped(2), vec![2, 2]);
        assert_eq!(grouped.free.in_order(), [G, ph, pw, B, oh, ow, CI]);
        assert_eq!(grouped.rhs, [G, ph, pw, rh, rw, CO, CI]);
        assert_eq!(grouped.out_axes(), [G, B, oh, ow, CI]);
    }
}
