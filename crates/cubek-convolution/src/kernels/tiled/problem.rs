use super::{ConvGather, ConvRoles};
use crate::components::ConvolutionOperation;
use crate::definition::ConvCost;
use crate::launch::SpatialWindow;
use cubecl::ir::ElemType;
use cubecl::tile::layout::{ElemAlignment, PhysicalAxisMap};
use cubecl::tile::{Axis, Projection, Space};
use cubecl::tune::Work;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_microkernel::matmul::{
    MatmulDestination, MatmulOperands, MatmulReading, MatmulRows, MatmulWorkItem,
};
use cubek_std::routine::EstimatedProblem;
use serde::{Deserialize, Serialize};

pub const B: Axis = Axis(0);
pub const CO: Axis = Axis(1);
pub const CI: Axis = Axis(2);
pub const OH: Axis = out(0);
pub const OW: Axis = out(1);
pub const RH: Axis = tap(0);
pub const RW: Axis = tap(1);

/// The output axis of spatial dim `dim`.
pub const fn out(dim: usize) -> Axis {
    Axis(3 + 2 * dim as u8)
}

/// The tap axis of spatial dim `dim`, which the window slides along with [`out`].
pub const fn tap(dim: usize) -> Axis {
    Axis(4 + 2 * dim as u8)
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
        views: MatmulOperands<[usize; 4]>,
        elements: MatmulOperands<ElemType>,
        sum: ElemType,
        alignments: MatmulOperands<ElemAlignment>,
        start_bytes: MatmulOperands<u64>,
    ) -> Self {
        let lhs = views.lhs;
        let extents = ConvRoles::of(operation)
            .expect("a problem is stated by tensors that were checked")
            .extents;
        Self {
            operation,
            batch: lhs[0],
            channels_in: extents.channels_in.read(&views),
            channels_out: extents.channels_out.read(&views),
            spatial: (0..window.len())
                .map(|dim| ConvDim {
                    gathered: lhs[dim + 1],
                    kernel: extents.kernel[dim].read(&views),
                    out: extents.out[dim].read(&views),
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
        ConvRoles::of(self.operation).expect("a problem is stated by tensors that were checked")
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
            axes: self.roles().axes(),
            reading: MatmulReading::AsStored,
        }
    }

    /// The seven axes of the contraction, in the order the operation lists them.
    pub fn space(&self) -> Space {
        Space::new(&self.roles().space.map(|axis| (axis, self.extent(axis))))
    }

    /// What `axis` measures in this problem.
    pub fn extent(&self, axis: Axis) -> usize {
        match axis {
            B => self.batch,
            OH => self.spatial[0].out,
            OW => self.spatial[1].out,
            CO => self.channels_out,
            RH => self.spatial[0].kernel,
            RW => self.spatial[1].kernel,
            CI => self.channels_in,
            _ => panic!("{axis:?} is not an axis of a convolution"),
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
            || self.spatial.iter().any(|dim| dim.window.padding > 0)
    }

    /// The input as the tiled contraction reads it, over [`ConvRoles::lhs`]: an NHWC buffer whose
    /// two spatial dims are each addressed by an output axis and a tap axis. The data backward
    /// reads the gradient the other way round, at stride 1.
    pub fn input_projection(&self) -> Projection {
        let roles = self.roles();
        let [[oh, rh], [ow, rw]] = ConvRoles::SPATIAL;
        Projection::new(
            &roles.lhs,
            &[
                PhysicalAxisMap::of(B),
                roles.gather.map(&self.spatial[0].window, oh, rh),
                roles.gather.map(&self.spatial[1].window, ow, rw),
                PhysicalAxisMap::of(roles.inner.lhs),
            ],
        )
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
        let (spatial_in, spatial_out) = self
            .spatial
            .iter()
            .map(|dim| gather.window_ends(dim))
            .unzip();
        ConvCost {
            batch: self.batch,
            channels_in: self.channels_in,
            spatial_in,
            channels_out: self.channels_out,
            kernel: self.spatial.iter().map(|dim| dim.kernel).collect(),
            spatial_out,
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
