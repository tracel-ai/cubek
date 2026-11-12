use super::ConvRoles;
use crate::components::ConvolutionOperation;
use crate::definition::Conv2dCost;
use crate::launch::ConvolutionArgs;
use cubecl::ir::ElemType;
use cubecl::tile::layout::{ElemAlignment, PhysicalAxisMap};
use cubecl::tile::{Axis, Projection, Space};
use cubecl::tune::Work;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_microkernel::matmul::{
    MatmulDestination, MatmulOperands, MatmulReading, MatmulRows, MatmulWorkItem,
};
use cubek_std::routine::EstimatedProblem;

pub const B: Axis = Axis(0);
pub const OH: Axis = Axis(1);
pub const OW: Axis = Axis(2);
pub const CO: Axis = Axis(3);
pub const RH: Axis = Axis(4);
pub const RW: Axis = Axis(5);
pub const CI: Axis = Axis(6);

/// A convolution with no groups and no bias, as the host describes it. Operands are read as a
/// matmul's: the input is the lhs.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvProblem {
    pub operation: ConvolutionOperation,
    pub batch: usize,
    pub channels_in: usize,
    /// The spatial extent of the gathered operand: the input, or the output gradient in the data
    /// backward.
    pub gathered: [usize; 2],
    pub channels_out: usize,
    pub kernel: [usize; 2],
    /// The output's spatial extent.
    pub out: [usize; 2],
    pub window: ConvolutionArgs<2>,
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
        window: ConvolutionArgs<2>,
        views: MatmulOperands<[usize; 4]>,
        elements: MatmulOperands<ElemType>,
        sum: ElemType,
        buffers: (MatmulOperands<ElemAlignment>, MatmulOperands<u64>),
    ) -> Self {
        let MatmulOperands { lhs, rhs, out } = views;
        let (channels_in, channels_out, kernel, out_extent) = match operation {
            ConvolutionOperation::Forward => (lhs[3], rhs[0], [rhs[1], rhs[2]], [out[1], out[2]]),
            ConvolutionOperation::BackwardWeight => {
                (lhs[3], out[3], [out[0], out[1]], [rhs[1], rhs[2]])
            }
            ConvolutionOperation::BackwardData | ConvolutionOperation::ForwardTransposed => {
                (out[3], lhs[3], [rhs[0], rhs[1]], [out[1], out[2]])
            }
        };
        let (alignments, start_bytes) = buffers;
        Self {
            operation,
            batch: lhs[0],
            channels_in,
            gathered: [lhs[1], lhs[2]],
            channels_out,
            kernel,
            out: out_extent,
            window,
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
            OH => self.out[0],
            OW => self.out[1],
            CO => self.channels_out,
            RH => self.kernel[0],
            RW => self.kernel[1],
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
        self.operation == ConvolutionOperation::BackwardData
            || self.window.padding.iter().any(|&padding| padding > 0)
    }

    /// The input as the tiled contraction reads it, over [`ConvRoles::lhs`]: an NHWC buffer whose
    /// two spatial dims are each addressed by an output axis and a tap axis. The data backward
    /// reads the gradient the other way round, at stride 1.
    pub fn input_projection(&self) -> Projection {
        let roles = self.roles();
        let window = &self.window;
        let along = |out, tap, dim| match self.operation {
            ConvolutionOperation::BackwardData => window.gradient_position_map(dim, out, tap),
            _ => window.position_map(dim, out, tap),
        };
        Projection::new(
            &roles.lhs,
            &[
                PhysicalAxisMap::of(B),
                along(OH, RH, 0),
                along(OW, RW, 1),
                PhysicalAxisMap::of(roles.inner.lhs),
            ],
        )
    }

    /// Whether every position the window reads lies in the input, so no read needs a check of its
    /// own.
    pub fn reads_inside(&self) -> bool {
        self.operation != ConvolutionOperation::BackwardData
            && (0..2).all(|dim| {
                let window = &self.window;
                let last = (self.out[dim].saturating_sub(1)) * window.stride[dim]
                    + self.kernel[dim].saturating_sub(1) * window.dilation[dim];
                window.padding[dim] == 0 && last < self.gathered[dim]
            })
    }
}

impl EstimatedProblem for ConvProblem {
    fn demand(&self) -> Work {
        // The data backward is the forward convolution of the gradient's shape over the input's.
        let (spatial_in, spatial_out) = match self.operation {
            ConvolutionOperation::BackwardData => (self.out, self.gathered),
            ConvolutionOperation::Forward
            | ConvolutionOperation::BackwardWeight
            | ConvolutionOperation::ForwardTransposed => (self.gathered, self.out),
        };
        Conv2dCost {
            batch: self.batch,
            channels_in: self.channels_in,
            spatial_in,
            channels_out: self.channels_out,
            kernel: self.kernel,
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
