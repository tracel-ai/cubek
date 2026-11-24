use super::{ConvGather, ConvGrouping, ConvPhasing, ConvRoles, SpatialDim};
use crate::components::ConvolutionOperation;
use crate::definition::{ConvCost, ConvCostDim};
use crate::launch::SpatialWindow;
use cubecl::ir::ElemType;
use cubecl::tile::layout::{ElemAlignment, PhysicalAxisMap};
use cubecl::tile::{Dimension, Projection, Space};
use cubecl::tune::Work;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_microkernel::matmul::{
    MatmulDestination, MatmulOperands, MatmulReading, MatmulRows, MatmulWorkItem,
};
use cubek_std::routine::EstimatedProblem;
use serde::{Deserialize, Serialize};

pub const B: Dimension = Dimension(0);
pub const CO: Dimension = Dimension(1);
pub const CI: Dimension = Dimension(2);
/// The group a channel belongs to, a dimension only a grouped convolution has.
pub const G: Dimension = Dimension(3);

/// How many spatial dims a convolution has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialDimCount(pub usize);

/// What a dimension of the space measures: the one place that numbers the labels of the dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvDimension {
    Batch,
    ChannelsOut,
    ChannelsIn,
    Groups,
    /// The output dimension of a spatial dim.
    Out(usize),
    /// The tap dimension of a spatial dim, which the window slides along with its output dimension.
    Tap(usize),
    /// The phase of a strided spatial dim in a data backward: which of the stride's residues of the
    /// tap and of the input position a cell belongs to.
    Phase(usize),
}

impl ConvDimension {
    /// The label of the first spatial dimension, which follows the three channel and batch labels.
    const FIRST_SPATIAL: u8 = G.0 + 1;
    /// A spatial dim takes three labels: its out, tap and phase dimensions.
    const PER_SPATIAL_DIM: u8 = 3;
    /// The most spatial dims whose labels, the last tap dimension included, fit in a label.
    pub const MAX_SPATIAL_DIMS: usize =
        ((u8::MAX - Self::FIRST_SPATIAL) / Self::PER_SPATIAL_DIM) as usize;

    /// The dimension of this measure, the label [`Self::of`] reads back.
    pub const fn dimension(self) -> Dimension {
        match self {
            Self::Batch => B,
            Self::ChannelsOut => CO,
            Self::ChannelsIn => CI,
            Self::Groups => G,
            Self::Out(dim) => Dimension(Self::FIRST_SPATIAL + Self::PER_SPATIAL_DIM * dim as u8),
            Self::Tap(dim) => {
                Dimension(Self::FIRST_SPATIAL + Self::PER_SPATIAL_DIM * dim as u8 + 1)
            }
            Self::Phase(dim) => {
                Dimension(Self::FIRST_SPATIAL + Self::PER_SPATIAL_DIM * dim as u8 + 2)
            }
        }
    }

    /// The measure of `dimension`, a label [`Self::dimension`] made.
    pub fn of(dimension: Dimension) -> Self {
        match dimension {
            B => Self::Batch,
            CO => Self::ChannelsOut,
            CI => Self::ChannelsIn,
            G => Self::Groups,
            Dimension(label) if label >= Self::FIRST_SPATIAL => Self::of_spatial(label),
            _ => panic!("{dimension:?} is not a dimension of a convolution"),
        }
    }

    /// The `[out, tap]` dimension pair of each of `spatial_dims` spatial dims.
    pub fn spatial(spatial_dims: SpatialDimCount) -> impl Iterator<Item = [Dimension; 2]> {
        (0..spatial_dims.0).map(|dim| [Self::Out(dim).dimension(), Self::Tap(dim).dimension()])
    }
}

impl ConvDimension {
    /// The measure of a spatial `label`.
    fn of_spatial(label: u8) -> Self {
        let offset = label - Self::FIRST_SPATIAL;
        let dim = usize::from(offset / Self::PER_SPATIAL_DIM);
        match offset % Self::PER_SPATIAL_DIM {
            0 => Self::Out(dim),
            1 => Self::Tap(dim),
            _ => Self::Phase(dim),
        }
    }
}

/// Whether a forward convolution adds a value to each output channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConvBias {
    /// The output is the sum alone.
    Absent,
    /// One value for each output channel, added to every cell of that channel.
    PerChannel,
}

/// A convolution of `groups` blocks of channels, each a convolution of its own, and at most one bias
/// value a channel, as the host describes it. Operands are read as a matmul's: the input is the lhs.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvProblem {
    pub operation: ConvolutionOperation,
    pub batch: usize,
    /// The blocks of channels, one for an ungrouped convolution.
    pub groups: usize,
    /// The input channels of one group.
    pub channels_in: usize,
    /// The output channels of one group.
    pub channels_out: usize,
    /// The extents and the window of each spatial dim.
    pub spatial: Vec<SpatialDim>,
    /// The element each operand is stored in.
    pub elements: MatmulOperands<ElemType>,
    /// The element the contraction sums in.
    pub sum: ElemType,
    /// The alignment of each operand's buffer.
    pub alignments: MatmulOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: MatmulOperands<u64>,
    /// Whether each output channel takes a value after the sum.
    pub bias: ConvBias,
}

impl ConvProblem {
    /// The problem of `operation` over operands whose shapes, as the roles of `grouping` view them
    /// ([`ConvRoles::views`]), are `views`.
    pub fn from_views(
        operation: ConvolutionOperation,
        grouping: ConvGrouping,
        window: &[SpatialWindow],
        views: MatmulOperands<Vec<usize>>,
        elements: MatmulOperands<ElemType>,
        sum: ElemType,
        alignments: MatmulOperands<ElemAlignment>,
        start_bytes: MatmulOperands<u64>,
    ) -> Self {
        let phasing = ConvPhasing::of(operation, window.iter().copied());
        let extents = ConvRoles::of(operation, SpatialDimCount(window.len()), grouping, phasing)
            .expect("a problem is stated by tensors that were checked")
            .extents;
        let (kernel, out) = (extents.kernel.read(&views), extents.out.read(&views));
        let gathered = extents.gathered.read(&views);
        Self {
            operation,
            batch: extents.batch.read(&views),
            groups: extents.groups.map_or(1, |groups| groups.read(&views)),
            channels_in: extents.channels_in.read(&views),
            channels_out: extents.channels_out.read(&views),
            spatial: (0..window.len())
                .map(|dim| SpatialDim {
                    gathered: gathered[dim],
                    kernel: kernel[dim],
                    out: out[dim],
                    window: window[dim],
                })
                .collect(),
            elements,
            sum,
            alignments,
            start_bytes,
            bias: ConvBias::Absent,
        }
    }

    /// The roles the operation gives the dimensions of the space.
    ///
    /// # Panics
    ///
    /// On a transposed forward, which [`ConvTensors`](super::ConvTensors) refuses before any
    /// problem is stated.
    pub fn roles(&self) -> ConvRoles {
        let grouping = ConvGrouping::of(self.groups);
        ConvRoles::of(
            self.operation,
            SpatialDimCount(self.spatial.len()),
            grouping,
            self.phasing(),
        )
        .expect("a problem is stated by tensors that were checked")
    }

    /// Whether the strided dims of this problem are split into phases.
    pub fn phasing(&self) -> ConvPhasing {
        ConvPhasing::of(
            self.operation,
            self.spatial.iter().map(|spatial_dim| spatial_dim.window),
        )
    }

    /// What every microkernel of this problem is handed, each operand's layout as its buffer
    /// lies ([`ConvRoles::layouts`]).
    pub fn work_item(&self) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: self.elements,
            sum: self.sum,
            layouts: self.roles().layouts(),
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            dimensions: self.roles().dimensions(),
            reading: MatmulReading::AsStored,
        }
    }

    /// The dimensions of the contraction, in the order the operation lists them.
    pub fn space(&self) -> Space {
        let dimensions: Vec<_> = (self.roles().space.into_iter())
            .map(|dimension| (dimension, self.extent(dimension)))
            .collect();
        Space::new(&dimensions)
    }

    /// The cells of the output's outer dimensions: the boxes of the output that a launch hands out
    /// one to each of its batches.
    pub fn outer_extent(&self) -> usize {
        let outer = self.roles().free.outer;
        outer
            .into_iter()
            .map(|dimension| self.extent(dimension))
            .product()
    }

    /// What `dimension` measures in this problem.
    pub fn extent(&self, dimension: Dimension) -> usize {
        let phasing = self.phasing();
        match ConvDimension::of(dimension) {
            ConvDimension::Batch => self.batch,
            ConvDimension::ChannelsOut => self.channels_out,
            ConvDimension::ChannelsIn => self.channels_in,
            ConvDimension::Groups => self.groups,
            ConvDimension::Out(dim) if phasing.is_phased(dim) => self.spatial[dim].phased_out(),
            ConvDimension::Out(dim) => self.spatial[dim].out,
            ConvDimension::Tap(dim) if phasing.is_phased(dim) => self.spatial[dim].phased_taps(),
            ConvDimension::Tap(dim) => self.spatial[dim].kernel,
            ConvDimension::Phase(dim) => self.spatial[dim].window.stride,
        }
    }

    /// Whether some contracted dimension has no cell, so every element of the output is a sum of
    /// nothing.
    pub fn contracts_nothing(&self) -> bool {
        let roles = self.roles();
        roles
            .contracted
            .in_order()
            .into_iter()
            .any(|dimension| self.extent(dimension) == 0)
    }

    /// The elements a kernel of this problem defines: the sum, the input and the output.
    pub fn kernel_elements(&self) -> [ElemType; 3] {
        [self.sum, self.elements.lhs, self.elements.out]
    }

    /// Whether a position can read before the input's start.
    pub fn underflows(&self) -> bool {
        self.roles().gather == ConvGather::Gradient
            || self
                .spatial
                .iter()
                .any(|spatial_dim| spatial_dim.window.padding > 0)
    }

    /// The input as the tiled contraction reads it, over [`ConvRoles::lhs`]: a channels last buffer,
    /// its channels split by group where the convolution is grouped, whose spatial dims are each
    /// addressed by an output dimension and a tap dimension. The data backward reads the gradient
    /// the other way round, at stride 1.
    pub fn input_projection(&self) -> Projection {
        let roles = self.roles();
        let spatial = roles.spatial().zip(&self.spatial).enumerate().map(
            |(dim, ([out, tap], spatial_dim))| match roles.phasing.is_phased(dim) {
                true => spatial_dim.phased_gradient_map(out, tap),
                false => roles.gather.map(&spatial_dim.window, out, tap),
            },
        );
        let groups = roles
            .grouping
            .dimensions()
            .iter()
            .map(|&dimension| PhysicalAxisMap::of(dimension));
        let maps: Vec<_> = [PhysicalAxisMap::of(B)]
            .into_iter()
            .chain(spatial)
            .chain(groups)
            .chain([PhysicalAxisMap::of(roles.inner.lhs)])
            .collect();
        Projection::new(&roles.lhs, &maps)
    }

    /// The weight of a phased data backward over [`ConvRoles::rhs`], whose tap axes are each
    /// addressed by a tap dimension and a phase dimension where the dim is strided.
    pub fn weight_projection(&self) -> Projection {
        let roles = self.roles();
        let taps = self.spatial.iter().enumerate().map(|(dim, spatial_dim)| {
            let tap = ConvDimension::Tap(dim).dimension();
            match roles.phasing.phase(dim) {
                Some(phase) => spatial_dim.phased_tap_map(tap, phase),
                None => PhysicalAxisMap::of(tap),
            }
        });
        let maps: Vec<_> = (roles.grouping.dimensions().iter())
            .map(|&dimension| PhysicalAxisMap::of(dimension))
            .chain(taps)
            .chain([PhysicalAxisMap::of(CO), PhysicalAxisMap::of(CI)])
            .collect();
        Projection::new(&roles.rhs, &maps)
    }

    /// The input gradient of a phased data backward over [`ConvRoles::free`], whose spatial axes
    /// are each addressed by an out dimension and a phase dimension where the dim is strided.
    pub fn output_projection(&self) -> Projection {
        let roles = self.roles();
        let spatial = self.spatial.iter().enumerate().map(|(dim, spatial_dim)| {
            let out = ConvDimension::Out(dim).dimension();
            match roles.phasing.phase(dim) {
                Some(phase) => spatial_dim.phased_out_map(out, phase),
                None => PhysicalAxisMap::of(out),
            }
        });
        let maps: Vec<_> = (roles.grouping.dimensions().iter())
            .map(|&dimension| PhysicalAxisMap::of(dimension))
            .chain([PhysicalAxisMap::of(B)])
            .chain(spatial)
            .chain([PhysicalAxisMap::of(CI)])
            .collect();
        Projection::new(&roles.free.in_order(), &maps)
    }

    /// The value of each tap map's launch coefficient ([`SpatialDim::phased_tap_map`]), the stride
    /// of each dim with a ragged phase, in the order of the weight's axes.
    pub fn tap_strides(&self) -> Vec<u32> {
        let phasing = self.phasing();
        (self.spatial.iter().enumerate())
            .filter(|&(dim, spatial_dim)| phasing.is_phased(dim) && spatial_dim.has_ragged_taps())
            .map(|(_, spatial_dim)| spatial_dim.window.stride as u32)
            .collect()
    }

    /// Whether every position the window reads lies in the input, so no read needs a check of its
    /// own. Never for a phased problem, whose gradient read is not the forward window.
    pub fn reads_inside(&self) -> bool {
        let roles = self.roles();
        roles.gather == ConvGather::Window
            && roles.phasing == ConvPhasing::Whole
            && self.spatial.iter().all(SpatialDim::reads_inside)
    }
}

impl EstimatedProblem for ConvProblem {
    fn demand(&self) -> Work {
        // The data backward is the forward convolution of the gradient's shape over the input's.
        let gather = self.roles().gather;
        ConvCost {
            batch: self.batch,
            groups: self.groups,
            channels_in: self.channels_in,
            channels_out: self.channels_out,
            spatial: self
                .spatial
                .iter()
                .map(|spatial_dim| {
                    let (input, out) = gather.window_ends(spatial_dim);
                    ConvCostDim {
                        input,
                        kernel: spatial_dim.kernel,
                        out,
                    }
                })
                .collect(),
            bias_elems: match self.bias {
                ConvBias::Absent => 0,
                ConvBias::PerChannel => self.groups * self.channels_out,
            },
            elems: MatmulGlobalElems {
                lhs: self.elements.lhs,
                rhs: self.elements.rhs,
                out: self.elements.out,
            },
        }
        .work()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem};
    use crate::launch::ConvolutionArgs;
    use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};

    #[test]
    fn every_dimension_reads_back_as_the_measure_that_made_it() {
        for count in 1..=3 {
            let spatial = (0..count).flat_map(|dim| {
                [
                    ConvDimension::Out(dim),
                    ConvDimension::Tap(dim),
                    ConvDimension::Phase(dim),
                ]
            });
            let measures = [
                ConvDimension::Batch,
                ConvDimension::ChannelsOut,
                ConvDimension::ChannelsIn,
                ConvDimension::Groups,
            ]
            .into_iter()
            .chain(spatial);
            let mut labels = Vec::new();
            for measure in measures {
                assert_eq!(ConvDimension::of(measure.dimension()), measure);
                labels.push(measure.dimension().0);
            }
            labels.sort_unstable();
            labels.dedup();
            assert_eq!(labels.len(), 4 + 3 * count);
        }
    }

    /// Each operation reads the batch, the channels and the spatial extents from the dims of its
    /// own operands: the data backward gathers the output gradient and writes the input's shape.
    #[test]
    fn extents_come_from_the_views_each_operation_states() {
        for operation in [Forward, BackwardWeight, BackwardData] {
            let problem = ConvProblem::from(&Conv2dProblem {
                input_shape: [2, 3, 10, 8],
                weight_shape: [5, 3, 3, 2],
                args: ConvolutionArgs {
                    stride: [1; 2],
                    padding: [0; 2],
                    dilation: [1; 2],
                },
                operation,
                element: Conv2dElement::F32,
                bias: ConvBias::Absent,
            });
            let spatial: Vec<_> = (problem.spatial.iter())
                .map(|spatial_dim| (spatial_dim.gathered, spatial_dim.kernel, spatial_dim.out))
                .collect();
            let expected = match operation {
                BackwardData => [(8, 3, 10), (7, 2, 8)],
                Forward | BackwardWeight | ConvolutionOperation::ForwardTransposed => {
                    [(10, 3, 8), (8, 2, 7)]
                }
            };
            let channels = (problem.batch, problem.channels_in, problem.channels_out);
            assert_eq!(channels, (2, 3, 5), "{operation:?}");
            assert_eq!(spatial, expected, "{operation:?}");
        }
    }
}
