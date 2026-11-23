use super::{ConvGather, ConvRoles};
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

/// How many spatial dims a convolution has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialDimCount(pub usize);

/// What an axis of the space measures: the one place that numbers the labels of the axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvAxis {
    Batch,
    ChannelsOut,
    ChannelsIn,
    /// The output axis of a spatial dim.
    Out(usize),
    /// The tap axis of a spatial dim, which the window slides along with its output axis.
    Tap(usize),
}

impl ConvAxis {
    /// The label of the first spatial axis, which follows the three channel and batch labels.
    const FIRST_SPATIAL: u8 = CI.0 + 1;
    /// A spatial dim takes two labels: its out axis, then its tap axis.
    const PER_SPATIAL_DIM: u8 = 2;
    /// The most spatial dims whose labels, the last tap axis included, fit in a label.
    pub const MAX_SPATIAL_DIMS: usize =
        ((u8::MAX - Self::FIRST_SPATIAL) / Self::PER_SPATIAL_DIM) as usize;

    /// The axis of this measure, the label [`Self::of`] reads back.
    pub const fn axis(self) -> Dimension {
        match self {
            Self::Batch => B,
            Self::ChannelsOut => CO,
            Self::ChannelsIn => CI,
            Self::Out(dim) => Dimension(Self::FIRST_SPATIAL + Self::PER_SPATIAL_DIM * dim as u8),
            Self::Tap(dim) => {
                Dimension(Self::FIRST_SPATIAL + Self::PER_SPATIAL_DIM * dim as u8 + 1)
            }
        }
    }

    /// The measure of `axis`, a label [`Self::axis`] made.
    pub fn of(axis: Dimension) -> Self {
        match axis {
            B => Self::Batch,
            CO => Self::ChannelsOut,
            CI => Self::ChannelsIn,
            Dimension(label) if label >= Self::FIRST_SPATIAL => Self::of_spatial(label),
            _ => panic!("{axis:?} is not an axis of a convolution"),
        }
    }

    /// The measure of a spatial `label`.
    fn of_spatial(label: u8) -> Self {
        let offset = label - Self::FIRST_SPATIAL;
        let dim = usize::from(offset / Self::PER_SPATIAL_DIM);
        if offset.is_multiple_of(Self::PER_SPATIAL_DIM) {
            Self::Out(dim)
        } else {
            Self::Tap(dim)
        }
    }

    /// The `[out, tap]` axis pair of each of `spatial_dims` spatial dims.
    pub fn spatial(spatial_dims: SpatialDimCount) -> impl Iterator<Item = [Dimension; 2]> {
        (0..spatial_dims.0).map(|dim| [Self::Out(dim).axis(), Self::Tap(dim).axis()])
    }
}

/// One spatial dim of a convolution: the extents it reads and writes along it and the window
/// between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConvDim {
    /// The extent of the gathered operand: the input, or the output gradient in the data backward.
    pub gathered: usize,
    pub kernel: usize,
    /// The output's extent.
    pub out: usize,
    pub window: SpatialWindow,
}

impl ConvDim {
    /// Whether every position the window reads lies in the gathered operand, so no read needs a
    /// check of its own.
    pub fn reads_inside(&self) -> bool {
        let last = self.out.saturating_sub(1) * self.window.stride
            + self.kernel.saturating_sub(1) * self.window.dilation;
        self.window.padding == 0 && last < self.gathered
    }
}

/// A convolution with no groups and no bias, as the host describes it. Operands are read as a
/// matmul's: the input is the lhs.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvProblem {
    pub operation: ConvolutionOperation,
    pub batch: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    /// The extents and the window of each spatial dim.
    pub spatial: Vec<ConvDim>,
    /// The element each operand is stored in.
    pub elements: MatmulOperands<ElemType>,
    /// The element the contraction sums in.
    pub sum: ElemType,
    /// The alignment of each operand's buffer.
    pub alignments: MatmulOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: MatmulOperands<u64>,
}

impl ConvProblem {
    /// The problem of `operation` over operands whose shapes, as the roles view them
    /// ([`ConvRoles::views`]), are `views`.
    pub fn from_views(
        operation: ConvolutionOperation,
        window: &[SpatialWindow],
        views: MatmulOperands<Vec<usize>>,
        elements: MatmulOperands<ElemType>,
        sum: ElemType,
        alignments: MatmulOperands<ElemAlignment>,
        start_bytes: MatmulOperands<u64>,
    ) -> Self {
        let extents = ConvRoles::of(operation, SpatialDimCount(window.len()))
            .expect("a problem is stated by tensors that were checked")
            .extents;
        let (kernel, out) = (extents.kernel.read(&views), extents.out.read(&views));
        let gathered = extents.gathered.read(&views);
        Self {
            operation,
            batch: extents.batch.read(&views),
            channels_in: extents.channels_in.read(&views),
            channels_out: extents.channels_out.read(&views),
            spatial: (0..window.len())
                .map(|dim| ConvDim {
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
        }
    }

    /// The roles the operation gives the axes of the space.
    ///
    /// # Panics
    ///
    /// On a transposed forward, which [`ConvTensors`](super::ConvTensors) refuses before any
    /// problem is stated.
    pub fn roles(&self) -> ConvRoles {
        ConvRoles::of(self.operation, SpatialDimCount(self.spatial.len()))
            .expect("a problem is stated by tensors that were checked")
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
            dimensions: self.roles().axes(),
            reading: MatmulReading::AsStored,
        }
    }

    /// The axes of the contraction, in the order the operation lists them.
    pub fn space(&self) -> Space {
        let axes: Vec<_> = (self.roles().space.into_iter())
            .map(|axis| (axis, self.extent(axis)))
            .collect();
        Space::new(&axes)
    }

    /// The cells of the output's outer axes: the boxes of the output that a launch hands out one
    /// to each of its batches.
    pub fn outer_extent(&self) -> usize {
        let outer = self.roles().free.outer;
        outer.into_iter().map(|axis| self.extent(axis)).product()
    }

    /// What `axis` measures in this problem.
    pub fn extent(&self, axis: Dimension) -> usize {
        match ConvAxis::of(axis) {
            ConvAxis::Batch => self.batch,
            ConvAxis::ChannelsOut => self.channels_out,
            ConvAxis::ChannelsIn => self.channels_in,
            ConvAxis::Out(dim) => self.spatial[dim].out,
            ConvAxis::Tap(dim) => self.spatial[dim].kernel,
        }
    }

    /// Whether some contracted axis has no cell, so every element of the output is a sum of
    /// nothing.
    pub fn contracts_nothing(&self) -> bool {
        let roles = self.roles();
        roles
            .contracted
            .in_order()
            .into_iter()
            .any(|axis| self.extent(axis) == 0)
    }

    /// The elements a kernel of this problem defines: the sum and the input.
    pub fn kernel_elements(&self) -> [ElemType; 2] {
        [self.sum, self.elements.lhs]
    }

    /// Whether a position can read before the input's start.
    pub fn underflows(&self) -> bool {
        self.roles().gather == ConvGather::Gradient
            || self
                .spatial
                .iter()
                .any(|conv_dim| conv_dim.window.padding > 0)
    }

    /// The input as the tiled contraction reads it, over [`ConvRoles::lhs`]: a channels last
    /// buffer whose spatial dims are each addressed by an output axis and a tap axis. The data
    /// backward reads the gradient the other way round, at stride 1.
    pub fn input_projection(&self) -> Projection {
        let roles = self.roles();
        let spatial = roles
            .spatial()
            .zip(&self.spatial)
            .map(|([out, tap], conv_dim)| roles.gather.map(&conv_dim.window, out, tap));
        let maps: Vec<_> = [PhysicalAxisMap::of(B)]
            .into_iter()
            .chain(spatial)
            .chain([PhysicalAxisMap::of(roles.inner.lhs)])
            .collect();
        Projection::new(&roles.lhs, &maps)
    }

    /// Whether every position the window reads lies in the input, so no read needs a check of its
    /// own.
    pub fn reads_inside(&self) -> bool {
        self.roles().gather == ConvGather::Window && self.spatial.iter().all(ConvDim::reads_inside)
    }
}

impl EstimatedProblem for ConvProblem {
    fn demand(&self) -> Work {
        // The data backward is the forward convolution of the gradient's shape over the input's.
        let gather = self.roles().gather;
        ConvCost {
            batch: self.batch,
            channels_in: self.channels_in,
            channels_out: self.channels_out,
            spatial: self
                .spatial
                .iter()
                .map(|conv_dim| {
                    let (input, out) = gather.window_ends(conv_dim);
                    ConvCostDim {
                        input,
                        kernel: conv_dim.kernel,
                        out,
                    }
                })
                .collect(),
            bias_elems: 0,
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
    fn every_axis_reads_back_as_the_measure_that_made_it() {
        for count in 1..=3 {
            let spatial = (0..count).flat_map(|dim| [ConvAxis::Out(dim), ConvAxis::Tap(dim)]);
            let measures = [ConvAxis::Batch, ConvAxis::ChannelsOut, ConvAxis::ChannelsIn]
                .into_iter()
                .chain(spatial);
            let mut labels = Vec::new();
            for measure in measures {
                assert_eq!(ConvAxis::of(measure.axis()), measure);
                labels.push(measure.axis().0);
            }
            labels.sort_unstable();
            labels.dedup();
            assert_eq!(labels.len(), 3 + 2 * count);
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
            });
            let spatial: Vec<_> = (problem.spatial.iter())
                .map(|conv_dim| (conv_dim.gathered, conv_dim.kernel, conv_dim.out))
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
