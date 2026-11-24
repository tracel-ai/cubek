use super::{B, CI, CO, ConvDecline, ConvDimension, G, SpatialDim, SpatialDimCount};
use crate::components::ConvolutionOperation;
use crate::launch::SpatialWindow;
use cubecl::prelude::TensorBinding;
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::{MatmulDimensions, MatmulLayout, MatmulOperand, MatmulOperands};

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
    fn split(dimensions: &[Dimension]) -> Self {
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
    fn split(dimensions: &[Dimension]) -> Self {
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

/// How the gathered input's positions are addressed from an output dimension and a tap dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvGather {
    /// The window of a convolution: `out * stride + tap * dilation - padding`.
    Window,
    /// The window read backward, at stride 1: `out + padding - tap * dilation`, which falls before
    /// the input's start at the first positions whatever the padding.
    Gradient,
}

impl ConvGather {
    /// The map of `window` over the dimensions `out_dimension` and `tap_dimension`.
    pub fn map(
        self,
        window: &SpatialWindow,
        out_dimension: Dimension,
        tap_dimension: Dimension,
    ) -> PhysicalAxisMap {
        match self {
            Self::Window => window.position_map(out_dimension, tap_dimension),
            Self::Gradient => window.gradient_position_map(out_dimension, tap_dimension),
        }
    }

    /// The window's input and output extents along `dim`.
    pub fn window_ends(self, spatial_dim: &SpatialDim) -> (usize, usize) {
        match self {
            Self::Window => (spatial_dim.gathered, spatial_dim.out),
            Self::Gradient => (spatial_dim.out, spatial_dim.gathered),
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
    const fn at(operand: MatmulOperand, dim: usize) -> Self {
        Self { operand, dim }
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
    pub batch: ViewDim,
    /// The group dim of the gathered input, where the channels are split by group.
    pub groups: Option<ViewDim>,
    /// The channels of the gathered input, whichever of the input and output channels they are.
    pub lhs_channels: ViewDim,
    /// The spatial extents of the gathered input.
    pub gathered: ViewRun,
    pub channels_in: ViewDim,
    pub channels_out: ViewDim,
    pub kernel: ViewRun,
    pub out: ViewRun,
}

/// Whether the channels are split into groups, each a convolution of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvGrouping {
    Whole,
    /// More than one group, as many as the count holds.
    Grouped(usize),
}

impl ConvGrouping {
    /// The grouping of a convolution of `groups` groups. A count of zero never reaches it:
    /// [`ConvChannelOperands::groups`] declines the operands that state one.
    pub fn of(groups: usize) -> Self {
        match groups {
            1 => Self::Whole,
            _ => Self::Grouped(groups),
        }
    }

    /// The dimensions this grouping adds ahead of the free ones: the group, where there is one.
    pub fn dimensions(self) -> &'static [Dimension] {
        match self {
            Self::Whole => &[],
            Self::Grouped(_) => &[G],
        }
    }
}

/// Where a tensor's channel axis is, the one a split by group cuts in two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAxis {
    /// A weight's filters, or a bias.
    First,
    /// An activation's channels.
    Last,
}

impl ChannelAxis {
    /// `tensor` with its channel axis cut into `[groups, channels / groups]` where `grouping` is
    /// grouped, by shape and strides alone. An axis the groups do not divide is the caller's to
    /// have refused.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::PermutedTiled`], where `grouping` is grouped and `tensor` is tiled.
    pub fn split(
        self,
        mut tensor: TensorBinding,
        grouping: ConvGrouping,
    ) -> Result<TensorBinding, Decline> {
        let ConvGrouping::Grouped(groups) = grouping else {
            return Ok(tensor);
        };
        if tensor.tiling.is_tiled() {
            return Err(ConvDecline::PermutedTiled.into());
        }
        let axis = match self {
            Self::First => 0,
            Self::Last => tensor.shape.len() - 1,
        };
        let (extent, stride) = (tensor.shape[axis], tensor.strides[axis]);
        let per_group = extent / groups;
        let (mut shape, mut strides) = (tensor.shape.to_vec(), tensor.strides.to_vec());
        shape.splice(axis..=axis, [groups, per_group]);
        strides.splice(axis..=axis, [stride * per_group, stride]);
        tensor.shape = Shape::from(shape);
        tensor.strides = Strides::new(&strides);
        Ok(tensor)
    }
}

/// Which operand of an operation holds the input channels, which the output channels and which is
/// the weight (or its gradient): where the group count is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvChannelOperands {
    /// The activation whose channels are the input's.
    pub input: MatmulOperand,
    /// The activation whose channels are the output's.
    pub output: MatmulOperand,
    /// The weight or its gradient, whose first dim is the filters and whose last is the input
    /// channels of one group.
    pub weight: MatmulOperand,
}

/// The channel counts a group count is read from.
struct GroupChannels {
    input: usize,
    output: usize,
    filters: usize,
    per_group: usize,
}

/// What the channel counts of a convolution state about its groups, the cases in the order they
/// are told apart.
#[derive(Debug, PartialEq, Eq)]
enum GroupSplit {
    /// No input channels and none per group: nothing to split.
    Empty,
    /// Input channels that are none, or not a whole number of the weight's, or a weight of none.
    Indivisible,
    /// Filters that the groups do not divide.
    FiltersIndivisible,
    /// Output channels that the groups do not divide.
    OutputIndivisible,
    /// Groups of one input and one output channel each.
    Depthwise,
    /// This many groups, one or more.
    Grouped(usize),
}

impl GroupSplit {
    fn of(channels: &GroupChannels) -> Self {
        let GroupChannels {
            input,
            output,
            filters,
            per_group,
        } = *channels;
        if per_group == 0 && input == 0 {
            return Self::Empty;
        }
        if per_group == 0 || input == 0 || input % per_group != 0 {
            return Self::Indivisible;
        }
        let groups = input / per_group;
        if filters % groups != 0 {
            return Self::FiltersIndivisible;
        }
        if output % groups != 0 {
            return Self::OutputIndivisible;
        }
        if groups > 1 && per_group == 1 && filters == groups {
            return Self::Depthwise;
        }
        Self::Grouped(groups)
    }
}

impl ConvChannelOperands {
    /// Where `operation` keeps its channels.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::Transposed`], for a transposed forward.
    pub fn of(operation: ConvolutionOperation) -> Result<Self, ConvDecline> {
        use MatmulOperand::{Lhs, Out, Rhs};
        match operation {
            ConvolutionOperation::Forward => Ok(Self {
                input: Lhs,
                output: Out,
                weight: Rhs,
            }),
            ConvolutionOperation::BackwardWeight => Ok(Self {
                input: Lhs,
                output: Rhs,
                weight: Out,
            }),
            ConvolutionOperation::BackwardData => Ok(Self {
                input: Out,
                output: Lhs,
                weight: Rhs,
            }),
            ConvolutionOperation::ForwardTransposed => Err(ConvDecline::Transposed),
        }
    }

    /// Where `operand`'s channel axis is.
    pub fn axis(&self, operand: MatmulOperand) -> ChannelAxis {
        match operand == self.weight {
            true => ChannelAxis::First,
            false => ChannelAxis::Last,
        }
    }

    /// The group count `shapes`, the operands in their own layouts, state: the input channels over
    /// the weight's channels per group.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::Groups`], where the input channels or the weight's filters are not whole
    /// groups of the weight's input channels. [`ConvDecline::OutChannels`], where the output's
    /// channels do not split into those groups. [`ConvDecline::Depthwise`], for a group of one input
    /// and one output channel.
    pub fn groups(&self, shapes: &MatmulOperands<&[usize]>) -> Result<usize, ConvDecline> {
        let channels = |operand| {
            shapes
                .get_ref(operand)
                .last()
                .copied()
                .expect("an operand has a channels dim")
        };
        let counts = GroupChannels {
            input: channels(self.input),
            output: channels(self.output),
            filters: shapes.get_ref(self.weight)[0],
            per_group: channels(self.weight),
        };
        match GroupSplit::of(&counts) {
            GroupSplit::Empty => Ok(1),
            GroupSplit::Indivisible | GroupSplit::FiltersIndivisible => Err(ConvDecline::Groups),
            GroupSplit::OutputIndivisible => Err(ConvDecline::OutChannels),
            GroupSplit::Depthwise => Err(ConvDecline::Depthwise),
            GroupSplit::Grouped(groups) => Ok(groups),
        }
    }
}

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
    fn view(
        self,
        spatial_dims: SpatialDimCount,
        grouping: ConvGrouping,
        values: &[usize],
    ) -> Vec<usize> {
        let buffer_dims = self.buffer_dims(spatial_dims, grouping);
        buffer_dims.into_iter().map(|dim| values[dim]).collect()
    }

    /// `tensor` with its dims listed in this order.
    fn permute(
        self,
        mut tensor: TensorBinding,
        spatial_dims: SpatialDimCount,
        grouping: ConvGrouping,
    ) -> Result<TensorBinding, Decline> {
        let buffer_dims = self.buffer_dims(spatial_dims, grouping);
        if buffer_dims.iter().copied().eq(0..tensor.shape.len()) {
            return Ok(tensor);
        }
        if tensor.tiling.is_tiled() {
            return Err(ConvDecline::PermutedTiled.into());
        }
        tensor.shape = Shape::from(self.view(spatial_dims, grouping, &tensor.shape));
        tensor.strides = Strides::new(&self.view(spatial_dims, grouping, &tensor.strides));
        Ok(tensor)
    }
}

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

/// The out and tap dimensions of a convolution over `spatial_dims` spatial dims, and where their
/// extents sit in the operands' views.
#[derive(Debug)]
struct SpatialDimensions {
    spatial_dims: SpatialDimCount,
    grouping: ConvGrouping,
    outs: Vec<Dimension>,
    taps: Vec<Dimension>,
}

/// The dimensions of `parts` one after the other; the slices let arrays and `Vec`s mix in one list.
fn joined(parts: &[&[Dimension]]) -> Vec<Dimension> {
    parts.concat()
}

impl SpatialDimensions {
    fn new(spatial_dims: SpatialDimCount, grouping: ConvGrouping) -> Self {
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
    fn split_dims(&self) -> usize {
        self.grouping.dimensions().len()
    }

    /// The channels dim of `operand`'s view, after the batch or filters, the groups and the
    /// spatial dims.
    fn channels(&self, operand: MatmulOperand) -> ViewDim {
        ViewDim::at(operand, self.spatial_dims.0 + 1 + self.split_dims())
    }

    /// The group dim of the gathered input, the lhs, which is split but never reordered.
    fn groups(&self) -> Option<ViewDim> {
        (self.split_dims() > 0).then(|| ViewDim::at(MatmulOperand::Lhs, self.spatial_dims.0 + 1))
    }

    /// The spatial dims of `operand`'s view, from `first`.
    fn run(&self, operand: MatmulOperand, first: usize) -> ViewRun {
        ViewRun {
            operand,
            first,
            len: self.spatial_dims.0,
        }
    }
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

    fn counts(input: usize, output: usize, filters: usize, per_group: usize) -> GroupChannels {
        GroupChannels {
            input,
            output,
            filters,
            per_group,
        }
    }

    #[test]
    fn group_split_tells_each_case_apart() {
        let cases = [
            (counts(0, 8, 8, 0), GroupSplit::Empty),
            (counts(0, 8, 8, 4), GroupSplit::Indivisible),
            (counts(8, 8, 8, 0), GroupSplit::Indivisible),
            (counts(10, 8, 8, 4), GroupSplit::Indivisible),
            (counts(8, 8, 9, 4), GroupSplit::FiltersIndivisible),
            (counts(8, 9, 8, 4), GroupSplit::OutputIndivisible),
            (counts(8, 8, 8, 1), GroupSplit::Depthwise),
            (counts(8, 16, 16, 1), GroupSplit::Grouped(8)),
            (counts(8, 8, 8, 8), GroupSplit::Grouped(1)),
            (counts(8, 6, 6, 4), GroupSplit::Grouped(2)),
        ];
        for (channels, expected) in cases {
            assert_eq!(GroupSplit::of(&channels), expected);
        }
    }

    #[test]
    fn groups_read_from_the_operands_in_their_own_layouts() {
        let forward = ConvChannelOperands::of(Forward).expect("a served operation");
        let shapes = |lhs, rhs, out| MatmulOperands { lhs, rhs, out };
        let grouped = shapes(&[1, 6, 6, 64][..], &[32, 3, 3, 16][..], &[1, 6, 6, 32][..]);
        assert_eq!(forward.groups(&grouped), Ok(4));
        let whole = shapes(&[1, 6, 6, 64][..], &[32, 3, 3, 64][..], &[1, 6, 6, 32][..]);
        assert_eq!(forward.groups(&whole), Ok(1));
        let weight = ConvChannelOperands::of(BackwardWeight).expect("a served operation");
        let grad = shapes(&[1, 6, 6, 64][..], &[1, 6, 6, 32][..], &[32, 3, 3, 16][..]);
        assert_eq!(weight.groups(&grad), Ok(4));
    }

    #[test]
    fn groups_decline_by_name() {
        let forward = ConvChannelOperands::of(Forward).expect("a served operation");
        let declined = |lhs: &[usize], rhs: &[usize], out: &[usize]| {
            forward.groups(&MatmulOperands { lhs, rhs, out })
        };
        assert_eq!(
            declined(&[1, 6, 6, 64], &[32, 3, 3, 24], &[1, 6, 6, 32]),
            Err(ConvDecline::Groups)
        );
        assert_eq!(
            declined(&[1, 6, 6, 64], &[30, 3, 3, 16], &[1, 6, 6, 30]),
            Err(ConvDecline::Groups)
        );
        assert_eq!(
            declined(&[1, 6, 6, 64], &[32, 3, 3, 16], &[1, 6, 6, 30]),
            Err(ConvDecline::OutChannels)
        );
        assert_eq!(
            declined(&[1, 6, 6, 16], &[16, 3, 3, 1], &[1, 6, 6, 16]),
            Err(ConvDecline::Depthwise)
        );
    }

    #[test]
    fn a_group_count_names_its_grouping() {
        assert_eq!(ConvGrouping::of(1), ConvGrouping::Whole);
        assert_eq!(ConvGrouping::of(3), ConvGrouping::Grouped(3));
    }
}
