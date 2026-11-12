use super::{B, CI, CO, ConvDecline, ConvDim, out, tap};
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
    /// The map of `window` over the axes `out` and `tap`.
    pub fn map(self, window: &SpatialWindow, out: Axis, tap: Axis) -> PhysicalAxisMap {
        match self {
            Self::Window => window.position_map(out, tap),
            Self::Gradient => window.gradient_position_map(out, tap),
        }
    }

    /// The window's input and output extents along `dim`.
    pub fn window_ends(self, dim: &ConvDim) -> (usize, usize) {
        match self {
            Self::Window => (dim.gathered, dim.out),
            Self::Gradient => (dim.out, dim.gathered),
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
    /// The dim of a buffer over `spatial` spatial dims that each dim of its view reads.
    pub fn dims(self, spatial: usize) -> Vec<usize> {
        match self {
            Self::Stored => (0..spatial + 2).collect(),
            Self::ChannelsOutLast => (1..spatial + 2).chain([0]).collect(),
            Self::ChannelsOutAfterTaps => (1..=spatial).chain([0, spatial + 1]).collect(),
        }
    }

    /// `values`, one per dim of a buffer, in the order of its view.
    fn view(self, spatial: usize, values: &[usize]) -> Vec<usize> {
        let dims = self.dims(spatial);
        dims.into_iter().map(|dim| values[dim]).collect()
    }
}

/// Which axis of the space plays which part in an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvRoles {
    /// The spatial dims of the convolution.
    pub dims: usize,
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

impl ConvRoles {
    /// The roles `operation` gives the axes of the space of a convolution over `dims` spatial
    /// dims.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::NoSpatialDim`], for no spatial dim, and [`ConvDecline::Transposed`], for a
    /// transposed forward.
    pub fn of(operation: ConvolutionOperation, dims: usize) -> Result<Self, Refusal> {
        if dims == 0 {
            return Err(ConvDecline::NoSpatialDim.into());
        }
        let (outs, taps): (Vec<Axis>, Vec<Axis>) =
            (0..dims).map(|dim| (out(dim), tap(dim))).unzip();
        let join = |parts: &[&[Axis]]| parts.concat();
        use MatmulOperand::{Lhs, Out, Rhs};
        let channels = |operand| ViewDim::at(operand, dims + 1);
        let run = |operand, first| ViewRun {
            operand,
            first,
            len: dims,
        };
        let roles = match operation {
            ConvolutionOperation::Forward => {
                let free = join(&[&[B], &outs, &[CO]]);
                let contracted = join(&[&taps, &[CI]]);
                Self {
                    dims,
                    space: join(&[&free, &contracted]),
                    free: ConvFreeAxes::split(&free),
                    contracted: ConvContractedAxes::split(&contracted),
                    lhs: join(&[&[B], &outs, &taps, &[CI]]),
                    rhs: join(&[&[CO], &taps, &[CI]]),
                    rhs_order: ViewOrder::Stored,
                    out_order: ViewOrder::Stored,
                    inner: MatmulOperands {
                        lhs: CI,
                        rhs: CI,
                        out: CO,
                    },
                    gather: ConvGather::Window,
                    extents: ConvExtents {
                        channels_in: channels(Lhs),
                        channels_out: ViewDim::at(Rhs, 0),
                        kernel: run(Rhs, 1),
                        out: run(Out, 1),
                    },
                    extent_mismatch: ConvDecline::ForwardExtent,
                }
            }
            ConvolutionOperation::BackwardWeight => {
                let free = join(&[&taps, &[CI, CO]]);
                let contracted = join(&[&[B], &outs]);
                Self {
                    dims,
                    space: join(&[&contracted, &free]),
                    free: ConvFreeAxes::split(&free),
                    contracted: ConvContractedAxes::split(&contracted),
                    lhs: join(&[&taps, &[B], &outs, &[CI]]),
                    rhs: join(&[&[B], &outs, &[CO]]),
                    rhs_order: ViewOrder::Stored,
                    out_order: ViewOrder::ChannelsOutLast,
                    inner: MatmulOperands {
                        lhs: CI,
                        rhs: CO,
                        out: CI,
                    },
                    gather: ConvGather::Window,
                    extents: ConvExtents {
                        channels_in: channels(Lhs),
                        channels_out: channels(Out),
                        kernel: run(Out, 0),
                        out: run(Rhs, 1),
                    },
                    extent_mismatch: ConvDecline::WeightGradExtent,
                }
            }
            ConvolutionOperation::BackwardData => {
                let free = join(&[&[B], &outs, &[CI]]);
                let contracted = join(&[&taps, &[CO]]);
                Self {
                    dims,
                    space: join(&[&free, &contracted]),
                    free: ConvFreeAxes::split(&free),
                    contracted: ConvContractedAxes::split(&contracted),
                    lhs: join(&[&[B], &outs, &taps, &[CO]]),
                    rhs: join(&[&taps, &[CO, CI]]),
                    rhs_order: ViewOrder::ChannelsOutAfterTaps,
                    out_order: ViewOrder::Stored,
                    inner: MatmulOperands {
                        lhs: CO,
                        rhs: CI,
                        out: CI,
                    },
                    gather: ConvGather::Gradient,
                    extents: ConvExtents {
                        channels_in: channels(Out),
                        channels_out: channels(Lhs),
                        kernel: run(Rhs, 0),
                        out: run(Out, 1),
                    },
                    extent_mismatch: ConvDecline::DataGradExtent,
                }
            }
            ConvolutionOperation::ForwardTransposed => {
                return Err(ConvDecline::Transposed.into());
            }
        };
        Ok(roles)
    }

    /// The spatial `[out, tap]` axis pairs, one per spatial dim of the gathered input.
    pub fn spatial(&self) -> impl Iterator<Item = [Axis; 2]> {
        (0..self.dims).map(|dim| [out(dim), tap(dim)])
    }

    /// The axis a kernel leaves to its launch: the batch where it is a free axis, which no
    /// vectorization or overhang decision reads. The weight backward contracts it, and the tile
    /// DSL's gather reads the extents of the axes it gathers over at expansion, so it stays
    /// compiled in.
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
            rhs: self.rhs_order.view(self.dims, &shapes.rhs),
            out: self.out_order.view(self.dims, &shapes.out),
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
                &order.view(self.dims, shape),
                &order.view(self.dims, &strides),
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
        [&[free.m, free.n, free.across], &free.outer[..]].concat()
    }

    /// `rhs` with its dims listed in the order of [`Self::rhs`].
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the dims move and `rhs` is tiled.
    pub fn rhs_view(&self, rhs: TensorBinding) -> Result<TensorBinding, Refusal> {
        permuted(rhs, self.rhs_order, self.dims)
    }

    /// `out`, a tensor of the operation's output in its own layout, with its dims listed in the
    /// order of [`Self::free`].
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where the dims move and `out` is tiled.
    pub fn out_view(&self, out: TensorBinding) -> Result<TensorBinding, Refusal> {
        permuted(out, self.out_order, self.dims)
    }
}

fn permuted(
    mut tensor: TensorBinding,
    order: ViewOrder,
    spatial: usize,
) -> Result<TensorBinding, Refusal> {
    if order == ViewOrder::Stored {
        return Ok(tensor);
    }
    if tensor.tiling.is_tiled() {
        return Err(ConvDecline::PermutedTiled.into());
    }
    tensor.shape = Shape::from(order.view(spatial, &tensor.shape));
    tensor.strides = Strides::new(&order.view(spatial, &tensor.strides));
    Ok(tensor)
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
        let (oh, ow, rh, rw) = (out(0), out(1), tap(0), tap(1));
        let roles = |operation| ConvRoles::of(operation, 2).expect("a served operation");

        let forward = roles(Forward);
        assert_eq!(forward.free.in_order(), [B, oh, ow, CO]);
        assert_eq!(forward.contracted.in_order(), [rh, rw, CI]);
        assert_eq!(forward.space, [B, oh, ow, CO, rh, rw, CI]);
        assert_eq!(forward.lhs, [B, oh, ow, rh, rw, CI]);
        assert_eq!(forward.rhs, [CO, rh, rw, CI]);
        assert_eq!(forward.rhs_order.dims(2), [0, 1, 2, 3]);
        assert_eq!(forward.out_order.dims(2), [0, 1, 2, 3]);
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
        assert_eq!(weight.rhs_order.dims(2), [0, 1, 2, 3]);
        assert_eq!(weight.out_order.dims(2), [1, 2, 3, 0]);
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
        assert_eq!(data.rhs_order.dims(2), [1, 2, 0, 3]);
        assert_eq!(data.out_order.dims(2), [0, 1, 2, 3]);
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
            for dims in [1, 3] {
                let roles = ConvRoles::of(operation, dims).expect("a served operation");
                let space: HashSet<Axis> = roles.space.iter().copied().collect();
                assert_eq!(space.len(), roles.space.len(), "{operation:?} {dims}");
                assert_eq!(space.len(), 2 * dims + 3, "{operation:?} {dims}");
                let named: HashSet<Axis> = [&roles.lhs, &roles.rhs, &roles.free.in_order()]
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                assert_eq!(named, space, "{operation:?} {dims}");
                for order in [roles.rhs_order, roles.out_order] {
                    let mut listed = order.dims(dims);
                    listed.sort_unstable();
                    assert_eq!(listed, (0..dims + 2).collect::<Vec<_>>());
                }
            }
        }
    }
}
