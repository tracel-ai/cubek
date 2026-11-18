use super::{B, CI, CO, ConvAxis, ConvDecline, ConvDim};
use crate::components::ConvolutionOperation;
use crate::launch::SpatialWindow;
use cubecl::prelude::TensorBinding;
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::{MatmulAxes, MatmulLayout, MatmulOperand, MatmulOperands};

/// The output's axes by the part each plays. The outer axes and the one across take a tile each of
/// a launch's cubes or planes, after the rows and the columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvFreeAxes {
    pub outer: Vec<Axis>,
    pub across: Axis,
    pub m: Axis,
    pub n: Axis,
}

impl ConvFreeAxes {
    /// The last three of `axes` as the one across, the rows and the columns, the rest outer.
    fn split(axes: &[Axis]) -> Self {
        let [outer @ .., across, m, n] = axes else {
            panic!("a convolution's output has an across, a rows and a columns axis");
        };
        Self {
            outer: outer.to_vec(),
            across: *across,
            m: *m,
            n: *n,
        }
    }

    /// The axes in the order the instruction reads them.
    pub fn in_order(&self) -> Vec<Axis> {
        [&self.outer[..], &[self.across, self.m, self.n]].concat()
    }
}

/// The contracted axes a box walks whole. `beside` fills a step of the contraction where `k` alone
/// is shorter than the instruction's `k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvContractedAxes {
    pub outer: Vec<Axis>,
    pub beside: Axis,
    pub k: Axis,
}

impl ConvContractedAxes {
    /// The last two of `axes` as the one beside `k` and `k`, the rest outer.
    fn split(axes: &[Axis]) -> Self {
        let [outer @ .., beside, k] = axes else {
            panic!("a convolution contracts a beside and a k axis");
        };
        Self {
            outer: outer.to_vec(),
            beside: *beside,
            k: *k,
        }
    }

    /// The axes in the order a box walks them.
    pub fn in_order(&self) -> Vec<Axis> {
        [&self.outer[..], &[self.beside, self.k]].concat()
    }
}

/// How the gathered input's positions are addressed from an output axis and a tap axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvGather {
    /// The window of a convolution: `out * stride + tap * dilation - padding`.
    Window,
    /// The window read backward, at stride 1: `out + padding - tap * dilation`, which falls before
    /// the input's start at the first positions whatever the padding.
    Gradient,
}

impl ConvGather {
    /// The map of `window` over the axes `out_axis` and `tap_axis`.
    pub fn map(self, window: &SpatialWindow, out_axis: Axis, tap_axis: Axis) -> PhysicalAxisMap {
        match self {
            Self::Window => window.position_map(out_axis, tap_axis),
            Self::Gradient => window.gradient_position_map(out_axis, tap_axis),
        }
    }

    /// The window's input and output extents along `dim`.
    pub fn window_ends(self, conv_dim: &ConvDim) -> (usize, usize) {
        match self {
            Self::Window => (conv_dim.gathered, conv_dim.out),
            Self::Gradient => (conv_dim.out, conv_dim.gathered),
        }
    }
}

/// One dim of one operand's view, where a problem reads an extent from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewDim {
    operand: MatmulOperand,
    dim: usize,
}

impl ViewDim {
    const fn at(operand: MatmulOperand, dim: usize) -> Self {
        Self { operand, dim }
    }

    /// The extent of this dim in `views`.
    pub fn read(self, views: &MatmulOperands<Vec<usize>>) -> usize {
        views.get_ref(self.operand)[self.dim]
    }
}

/// The run of dims of one operand's view that hold the spatial extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewRun {
    operand: MatmulOperand,
    first: usize,
    len: usize,
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
    pub channels_in: ViewDim,
    pub channels_out: ViewDim,
    pub kernel: ViewRun,
    pub out: ViewRun,
}

/// The order a view lists the dims of an operand's buffer in, where the roles move the output
/// channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewOrder {
    Stored,
    /// The output channels, stored first, listed last.
    ChannelsOutLast,
    /// The output channels, stored first, listed just before the input channels.
    ChannelsOutAfterTaps,
}

impl ViewOrder {
    /// The dim of a buffer over `spatial_dims` spatial dims that each dim of its view reads.
    pub fn buffer_dims(self, spatial_dims: usize) -> Vec<usize> {
        match self {
            Self::Stored => (0..spatial_dims + 2).collect(),
            Self::ChannelsOutLast => (1..spatial_dims + 2).chain([0]).collect(),
            Self::ChannelsOutAfterTaps => (1..=spatial_dims).chain([0, spatial_dims + 1]).collect(),
        }
    }

    /// `values`, one per dim of a buffer, in the order of its view.
    fn view(self, spatial_dims: usize, values: &[usize]) -> Vec<usize> {
        let buffer_dims = self.buffer_dims(spatial_dims);
        buffer_dims.into_iter().map(|dim| values[dim]).collect()
    }

    /// `tensor` with its dims listed in this order.
    fn permute(
        self,
        mut tensor: TensorBinding,
        spatial_dims: usize,
    ) -> Result<TensorBinding, Refusal> {
        if self == Self::Stored {
            return Ok(tensor);
        }
        if tensor.tiling.is_tiled() {
            return Err(ConvDecline::PermutedTiled.into());
        }
        tensor.shape = Shape::from(self.view(spatial_dims, &tensor.shape));
        tensor.strides = Strides::new(&self.view(spatial_dims, &tensor.strides));
        Ok(tensor)
    }
}

/// Which axis of the space plays which part in an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvRoles {
    /// The spatial dims of the convolution.
    pub spatial_dims: usize,
    pub free: ConvFreeAxes,
    pub contracted: ConvContractedAxes,
    /// The axes of the space, the rows before the columns: the planes of a cube take them in the
    /// space's order.
    pub space: Vec<Axis>,
    /// The axes the gathered input is read over, the contracted run just before the free one: a
    /// matrix instruction takes its operands' axes in that order.
    pub lhs: Vec<Axis>,
    /// The axes the rhs is bound by, in the order the instruction reads them.
    pub rhs: Vec<Axis>,
    /// The order the rhs's view lists the dims of its buffer in.
    pub rhs_order: ViewOrder,
    /// The order the output's view lists the dims of its buffer in.
    pub out_order: ViewOrder,
    /// The axis each operand's buffer is contiguous along: the one its vectors run along.
    pub inner: MatmulOperands<Axis>,
    /// How the gathered input is addressed.
    pub gather: ConvGather,
    pub extents: ConvExtents,
    /// What the window's extents are declined with where they do not yield the output's.
    pub extent_mismatch: ConvDecline,
}

/// The out and tap axes of a convolution over `spatial_dims` spatial dims, and where their extents sit in
/// the operands' views.
#[derive(Debug)]
struct SpatialAxes {
    spatial_dims: usize,
    outs: Vec<Axis>,
    taps: Vec<Axis>,
}

/// The axes of `parts` one after the other.
fn joined(parts: &[&[Axis]]) -> Vec<Axis> {
    parts.concat()
}

impl SpatialAxes {
    fn new(spatial_dims: usize) -> Self {
        let (outs, taps) = ConvAxis::spatial(spatial_dims)
            .map(|[out, tap]| (out, tap))
            .unzip();
        Self {
            spatial_dims,
            outs,
            taps,
        }
    }

    /// The channels dim of `operand`'s view, after the batch or filters and the spatial dims.
    fn channels(&self, operand: MatmulOperand) -> ViewDim {
        ViewDim::at(operand, self.spatial_dims + 1)
    }

    /// The spatial dims of `operand`'s view, from `first`.
    fn run(&self, operand: MatmulOperand, first: usize) -> ViewRun {
        ViewRun {
            operand,
            first,
            len: self.spatial_dims,
        }
    }
}

impl ConvRoles {
    /// The roles `operation` gives the axes of the space of a convolution over `spatial_dims` spatial
    /// dims.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::NoSpatialDim`], for no spatial dim, and [`ConvDecline::Transposed`], for a
    /// transposed forward.
    pub fn of(operation: ConvolutionOperation, spatial_dims: usize) -> Result<Self, Refusal> {
        if spatial_dims == 0 {
            return Err(ConvDecline::NoSpatialDim.into());
        }
        let spatial = SpatialAxes::new(spatial_dims);
        match operation {
            ConvolutionOperation::Forward => Ok(Self::forward(&spatial)),
            ConvolutionOperation::BackwardWeight => Ok(Self::backward_weight(&spatial)),
            ConvolutionOperation::BackwardData => Ok(Self::backward_data(&spatial)),
            ConvolutionOperation::ForwardTransposed => Err(ConvDecline::Transposed.into()),
        }
    }

    fn forward(spatial: &SpatialAxes) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let free = joined(&[&[B], &spatial.outs, &[CO]]);
        let contracted = joined(&[&spatial.taps, &[CI]]);
        Self {
            spatial_dims: spatial.spatial_dims,
            space: joined(&[&free, &contracted]),
            free: ConvFreeAxes::split(&free),
            contracted: ConvContractedAxes::split(&contracted),
            lhs: joined(&[&[B], &spatial.outs, &spatial.taps, &[CI]]),
            rhs: joined(&[&[CO], &spatial.taps, &[CI]]),
            rhs_order: ViewOrder::Stored,
            out_order: ViewOrder::Stored,
            inner: MatmulOperands {
                lhs: CI,
                rhs: CI,
                out: CO,
            },
            gather: ConvGather::Window,
            extents: ConvExtents {
                channels_in: spatial.channels(Lhs),
                channels_out: ViewDim::at(Rhs, 0),
                kernel: spatial.run(Rhs, 1),
                out: spatial.run(Out, 1),
            },
            extent_mismatch: ConvDecline::ForwardExtent,
        }
    }

    fn backward_weight(spatial: &SpatialAxes) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let free = joined(&[&spatial.taps, &[CI, CO]]);
        let contracted = joined(&[&[B], &spatial.outs]);
        Self {
            spatial_dims: spatial.spatial_dims,
            space: joined(&[&contracted, &free]),
            free: ConvFreeAxes::split(&free),
            contracted: ConvContractedAxes::split(&contracted),
            lhs: joined(&[&spatial.taps, &[B], &spatial.outs, &[CI]]),
            rhs: joined(&[&[B], &spatial.outs, &[CO]]),
            rhs_order: ViewOrder::Stored,
            out_order: ViewOrder::ChannelsOutLast,
            inner: MatmulOperands {
                lhs: CI,
                rhs: CO,
                out: CI,
            },
            gather: ConvGather::Window,
            extents: ConvExtents {
                channels_in: spatial.channels(Lhs),
                channels_out: spatial.channels(Out),
                kernel: spatial.run(Out, 0),
                out: spatial.run(Rhs, 1),
            },
            extent_mismatch: ConvDecline::WeightGradExtent,
        }
    }

    fn backward_data(spatial: &SpatialAxes) -> Self {
        use MatmulOperand::{Lhs, Out, Rhs};
        let free = joined(&[&[B], &spatial.outs, &[CI]]);
        let contracted = joined(&[&spatial.taps, &[CO]]);
        Self {
            spatial_dims: spatial.spatial_dims,
            space: joined(&[&free, &contracted]),
            free: ConvFreeAxes::split(&free),
            contracted: ConvContractedAxes::split(&contracted),
            lhs: joined(&[&[B], &spatial.outs, &spatial.taps, &[CO]]),
            rhs: joined(&[&spatial.taps, &[CO, CI]]),
            rhs_order: ViewOrder::ChannelsOutAfterTaps,
            out_order: ViewOrder::Stored,
            inner: MatmulOperands {
                lhs: CO,
                rhs: CI,
                out: CI,
            },
            gather: ConvGather::Gradient,
            extents: ConvExtents {
                channels_in: spatial.channels(Out),
                channels_out: spatial.channels(Lhs),
                kernel: spatial.run(Rhs, 0),
                out: spatial.run(Out, 1),
            },
            extent_mismatch: ConvDecline::DataGradExtent,
        }
    }

    /// The spatial `[out, tap]` axis pairs, one per spatial dim of the gathered input.
    pub fn spatial(&self) -> impl Iterator<Item = [Axis; 2]> {
        ConvAxis::spatial(self.spatial_dims)
    }

    /// The axis a kernel leaves to its launch: the batch where it is a free axis. No vectorization
    /// decision reads it, and over one spatial dim, where it is the across axis, its overhang is
    /// read off the problem's own space. The weight backward contracts it, and the tile DSL's
    /// gather reads the extents of the axes it gathers over at expansion, so it stays compiled in.
    pub fn dynamic(&self) -> Option<Axis> {
        self.free.in_order().contains(&B).then_some(B)
    }

    /// How each operand's buffer lies against the product's own `[m, k]`, `[k, n]` and `[m, n]`:
    /// row where it runs along the second of them, col along the first.
    pub fn layouts(&self) -> MatmulOperands<MatmulLayout> {
        let axes = self.axes();
        let layout = |operand: MatmulOperand| {
            let along = axes.dim(self.inner.get(operand));
            MatmulLayout::along(along.map(|dim| MatmulAxes::MNK.of(dim)), operand)
        };
        MatmulOperands {
            lhs: layout(MatmulOperand::Lhs),
            rhs: layout(MatmulOperand::Rhs),
            out: layout(MatmulOperand::Out),
        }
    }

    /// The axes of the instruction's rows, columns and contracted cells.
    pub fn axes(&self) -> MatmulAxes {
        MatmulAxes {
            m: self.free.m,
            n: self.free.n,
            k: self.contracted.k,
        }
    }

    /// `shapes`, the operation's operands in their own layouts, as each is viewed through the roles
    /// ([`Self::rhs_view`], [`Self::out_view`]).
    pub fn views(&self, shapes: &MatmulOperands<Vec<usize>>) -> MatmulOperands<Vec<usize>> {
        MatmulOperands {
            lhs: shapes.lhs.clone(),
            rhs: self.rhs_order.view(self.spatial_dims, &shapes.rhs),
            out: self.out_order.view(self.spatial_dims, &shapes.out),
        }
    }

    /// The alignments of dense buffers of `shapes`, laid out as the operation's tensors are, once
    /// the rhs and the output are viewed through the roles ([`Self::rhs_view`],
    /// [`Self::out_view`]).
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
                &order.view(self.spatial_dims, shape),
                &order.view(self.spatial_dims, &strides),
            )
        };
        MatmulOperands {
            lhs: dense(&shapes.lhs, ViewOrder::Stored),
            rhs: dense(&shapes.rhs, self.rhs_order),
            out: dense(&shapes.out, self.out_order),
        }
    }

    /// Whether `rhs` and `out`, viewed through the roles, and the input's `extent` of `axis`
    /// (`None` where the input does not carry it) all give `axis` the same extent.
    pub fn agree_on(
        &self,
        axis: Axis,
        extent: Option<usize>,
        [rhs, out]: [&TensorBinding; 2],
    ) -> bool {
        let free = self.free.in_order();
        let named = [(&self.rhs[..], rhs), (&free[..], out)];
        let mut extents = extent
            .into_iter()
            .chain(named.iter().filter_map(|(axes, tensor)| {
                let dim = axes.iter().position(|&named| named == axis)?;
                Some(tensor.shape[dim])
            }));
        let first = extents.next();
        extents.all(|extent| Some(extent) == first)
    }

    /// The axes a unit launch lays its planes over: the rows, the columns, the one across and
    /// the outer ones.
    pub fn plane_axes(&self) -> Vec<Axis> {
        let free = &self.free;
        joined(&[&[free.m, free.n, free.across], &free.outer[..]])
    }

    /// `rhs` with its dims listed in the order of [`Self::rhs`].
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the dims move and `rhs` is tiled.
    pub fn rhs_view(&self, rhs: TensorBinding) -> Result<TensorBinding, Refusal> {
        self.rhs_order.permute(rhs, self.spatial_dims)
    }

    /// `out`, a tensor of the operation's output in its own layout, with its dims listed in the
    /// order of [`Self::free`].
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the dims move and `out` is tiled.
    pub fn out_view(&self, out: TensorBinding) -> Result<TensorBinding, Refusal> {
        self.out_order.permute(out, self.spatial_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};
    use std::collections::HashSet;

    const OPERATIONS: [ConvolutionOperation; 3] = [Forward, BackwardWeight, BackwardData];

    /// A transposed forward is no operation of the tiled convolution, which states it as the data
    /// backward at a stride.
    #[test]
    fn the_transposed_forward_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(ConvolutionOperation::ForwardTransposed, 2),
            Err(ConvDecline::Transposed.into())
        );
    }

    #[test]
    fn no_spatial_dim_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(Forward, 0),
            Err(ConvDecline::NoSpatialDim.into())
        );
    }

    /// Over two spatial dims the roles are the tables the tiled convolution stated for 2D alone.
    #[test]
    fn two_spatial_dims_state_the_two_dimensional_tables() {
        let [[oh, rh], [ow, rw]] =
            [0, 1].map(|dim| [ConvAxis::Out(dim).axis(), ConvAxis::Tap(dim).axis()]);
        let roles = |operation| ConvRoles::of(operation, 2).expect("a served operation");

        let forward = roles(Forward);
        assert_eq!(forward.free.in_order(), [B, oh, ow, CO]);
        assert_eq!(forward.contracted.in_order(), [rh, rw, CI]);
        assert_eq!(forward.space, [B, oh, ow, CO, rh, rw, CI]);
        assert_eq!(forward.lhs, [B, oh, ow, rh, rw, CI]);
        assert_eq!(forward.rhs, [CO, rh, rw, CI]);
        assert_eq!(forward.rhs_order.buffer_dims(2), [0, 1, 2, 3]);
        assert_eq!(forward.out_order.buffer_dims(2), [0, 1, 2, 3]);
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
        assert_eq!(weight.rhs_order.buffer_dims(2), [0, 1, 2, 3]);
        assert_eq!(weight.out_order.buffer_dims(2), [1, 2, 3, 0]);
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
        assert_eq!(data.rhs_order.buffer_dims(2), [1, 2, 0, 3]);
        assert_eq!(data.out_order.buffer_dims(2), [0, 1, 2, 3]);
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

    /// Whatever the spatial count, every axis of the space is named once, the operands' axes
    /// cover it, and a view lists each dim of its buffer once.
    #[test]
    fn roles_of_any_spatial_count_name_each_axis_once() {
        for operation in OPERATIONS {
            for spatial_dims in [1, 3] {
                let roles = ConvRoles::of(operation, spatial_dims).expect("a served operation");
                let space: HashSet<Axis> = roles.space.iter().copied().collect();
                assert_eq!(
                    space.len(),
                    roles.space.len(),
                    "{operation:?} {spatial_dims}"
                );
                assert_eq!(
                    space.len(),
                    2 * spatial_dims + 3,
                    "{operation:?} {spatial_dims}"
                );
                let named: HashSet<Axis> = [&roles.lhs, &roles.rhs, &roles.free.in_order()]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                assert_eq!(named, space, "{operation:?} {spatial_dims}");
                for order in [roles.rhs_order, roles.out_order] {
                    let mut listed = order.buffer_dims(spatial_dims);
                    listed.sort_unstable();
                    assert_eq!(listed, (0..spatial_dims + 2).collect::<Vec<_>>());
                }
            }
        }
    }
}
