use super::{ConvOperation, ConvWindow};
use crate::definition::Conv2dCost;
use cubecl::ir::ElemType;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubecl::tune::Work;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_microkernel::matmul::{
    MatmulDestination, MatmulLayout, MatmulOperands, MatmulReading, MatmulRows, MatmulWorkItem,
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
    pub operation: ConvOperation,
    pub batch: usize,
    pub channels_in: usize,
    /// The input's spatial extent.
    pub input: [usize; 2],
    pub channels_out: usize,
    pub kernel: [usize; 2],
    /// The output's spatial extent.
    pub out: [usize; 2],
    pub window: ConvWindow,
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
    /// What every microkernel of this problem is handed. The tiles are stated by the projections
    /// that bind them, so no operand lies in a buffer order of its own.
    pub fn work_item(&self) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: self.elements,
            sum: self.sum,
            layouts: MatmulOperands::all(MatmulLayout::Row),
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: self.operation.roles().axes(),
            reading: MatmulReading::AsStored,
        }
    }

    /// The seven axes of the contraction, in the order the operation lists them.
    pub fn space(&self) -> Space {
        Space::new(
            &self
                .operation
                .roles()
                .space
                .map(|axis| (axis, self.extent(axis))),
        )
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

    /// The elements a kernel of this problem defines: the sum and the input.
    pub fn kernel_elements(&self) -> [ElemType; 2] {
        [self.sum, self.elements.lhs]
    }

    /// Whether every position the window reads lies in the input, so no read needs a check of its
    /// own.
    pub fn reads_inside(&self) -> bool {
        self.operation != ConvOperation::BackwardData
            && (0..2).all(|dim| {
                let window = &self.window;
                let last = (self.out[dim].saturating_sub(1)) * window.stride[dim]
                    + (self.kernel[dim] - 1) * window.dilation[dim];
                window.padding[dim] == 0 && last < self.input[dim]
            })
    }

    /// A problem of dense NHWC operands of `element`, summed in `f32`, with no padding or
    /// dilation: what a routine's pure tests prepare against.
    #[cfg(test)]
    pub(crate) fn dense(
        batch: usize,
        input: [usize; 2],
        (channels_in, channels_out): (usize, usize),
        kernel: [usize; 2],
        stride: usize,
        element: ElemType,
    ) -> Self {
        let out = [0, 1].map(|dim| (input[dim] - kernel[dim]) / stride + 1);
        let aligned = |shape: [usize; 4]| {
            let strides = [
                shape[1] * shape[2] * shape[3],
                shape[2] * shape[3],
                shape[3],
                1,
            ];
            ElemAlignment::new(&shape, &strides)
        };
        Self {
            operation: ConvOperation::Forward,
            batch,
            channels_in,
            input,
            channels_out,
            kernel,
            out,
            window: ConvWindow {
                stride: [stride; 2],
                dilation: [1; 2],
                padding: [0; 2],
            },
            elements: MatmulOperands::all(element),
            sum: ElemType::Float(cubecl::ir::FloatKind::F32),
            alignments: MatmulOperands {
                lhs: aligned([batch, input[0], input[1], channels_in]),
                rhs: aligned([channels_out, kernel[0], kernel[1], channels_in]),
                out: aligned([batch, out[0], out[1], channels_out]),
            },
            start_bytes: MatmulOperands::all(0),
        }
    }
}

impl EstimatedProblem for ConvProblem {
    fn demand(&self) -> Work {
        // The data backward is the forward convolution of the gradient's shape over the input's.
        let (spatial_in, spatial_out) = match self.operation {
            ConvOperation::BackwardData => (self.out, self.input),
            _ => (self.input, self.out),
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
