use super::{B, CI, CO, ConvDeclined, OH, OW, RH, RW};
use crate::components::ConvolutionOperation;
use crate::launch::ConvolutionArgs;
use cubecl::prelude::TensorBinding;
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::{MatmulAxes, MatmulLayout, MatmulOperand, MatmulOperands};

/// The output's axes by the part each plays. The batch and the one across take a tile each of a
/// cube or a plane, after the rows and the columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvFreeAxes {
    pub batch: Axis,
    pub across: Axis,
    pub m: Axis,
    pub n: Axis,
}

impl ConvFreeAxes {
    /// The axes in the order the instruction reads them.
    pub fn in_order(&self) -> [Axis; 4] {
        [self.batch, self.across, self.m, self.n]
    }
}

/// The contracted axes a box walks whole. `beside` fills a step of the contraction where `k` alone
/// is shorter than the instruction's `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvContractedAxes {
    pub outer: Axis,
    pub beside: Axis,
    pub k: Axis,
}

impl ConvContractedAxes {
    /// The axes in the order a box walks them.
    pub fn in_order(&self) -> [Axis; 3] {
        [self.outer, self.beside, self.k]
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
    /// The map of spatial `dim` of `window` over the axes `out` and `tap`.
    pub fn map(
        self,
        window: &ConvolutionArgs<2>,
        dim: usize,
        out: Axis,
        tap: Axis,
    ) -> PhysicalAxisMap {
        match self {
            Self::Window => window.position_map(dim, out, tap),
            Self::Gradient => window.gradient_position_map(dim, out, tap),
        }
    }

    /// `(gathered, out)` as the window's input and output extents.
    pub fn window_ends(self, gathered: [usize; 2], out: [usize; 2]) -> ([usize; 2], [usize; 2]) {
        match self {
            Self::Window => (gathered, out),
            Self::Gradient => (out, gathered),
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
    pub fn read(self, views: &MatmulOperands<[usize; 4]>) -> usize {
        views.get(self.operand)[self.dim]
    }
}

/// Where each extent of a problem lives in the operands' views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvExtents {
    pub channels_in: ViewDim,
    pub channels_out: ViewDim,
    pub kernel: [ViewDim; 2],
    pub out: [ViewDim; 2],
}

/// Which axis of the space plays which part in an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvRoles {
    pub free: ConvFreeAxes,
    pub contracted: ConvContractedAxes,
    /// The seven axes of the space, the rows before the columns: the planes of a cube take them
    /// in the space's order.
    pub space: [Axis; 7],
    /// The axes the gathered input is read over, the contracted run just before the free one: a
    /// matrix instruction takes its operands' axes in that order.
    pub lhs: [Axis; 6],
    /// The axes the rhs is bound by, in the order the instruction reads them.
    pub rhs: [Axis; 4],
    /// The dim of the rhs's buffer each of `rhs` reads.
    pub rhs_dims: [usize; 4],
    /// The dim of the output's buffer each of `free` reads.
    pub out_dims: [usize; 4],
    /// The axis each operand's buffer is contiguous along: the one its vectors run along.
    pub inner: MatmulOperands<Axis>,
    /// How the gathered input is addressed.
    pub gather: ConvGather,
    pub extents: ConvExtents,
    /// What the window's extents are declined with where they do not yield the output's.
    pub extent_mismatch: ConvDeclined,
}

impl ConvRoles {
    const FORWARD: Self = Self {
        free: ConvFreeAxes {
            batch: B,
            across: OH,
            m: OW,
            n: CO,
        },
        contracted: ConvContractedAxes {
            outer: RH,
            beside: RW,
            k: CI,
        },
        space: [B, OH, OW, CO, RH, RW, CI],
        lhs: [B, OH, OW, RH, RW, CI],
        rhs: [CO, RH, RW, CI],
        rhs_dims: [0, 1, 2, 3],
        out_dims: [0, 1, 2, 3],
        inner: MatmulOperands {
            lhs: CI,
            rhs: CI,
            out: CO,
        },
        gather: ConvGather::Window,
        extents: ConvExtents {
            channels_in: ViewDim::at(MatmulOperand::Lhs, 3),
            channels_out: ViewDim::at(MatmulOperand::Rhs, 0),
            kernel: [
                ViewDim::at(MatmulOperand::Rhs, 1),
                ViewDim::at(MatmulOperand::Rhs, 2),
            ],
            out: [
                ViewDim::at(MatmulOperand::Out, 1),
                ViewDim::at(MatmulOperand::Out, 2),
            ],
        },
        extent_mismatch: ConvDeclined::ForwardExtent,
    };

    const WEIGHT_BACKWARD: Self = Self {
        free: ConvFreeAxes {
            batch: RH,
            across: RW,
            m: CI,
            n: CO,
        },
        contracted: ConvContractedAxes {
            outer: B,
            beside: OH,
            k: OW,
        },
        space: [B, OH, OW, RH, RW, CI, CO],
        lhs: [RH, RW, B, OH, OW, CI],
        rhs: [B, OH, OW, CO],
        rhs_dims: [0, 1, 2, 3],
        out_dims: [1, 2, 3, 0],
        inner: MatmulOperands {
            lhs: CI,
            rhs: CO,
            out: CI,
        },
        gather: ConvGather::Window,
        extents: ConvExtents {
            channels_in: ViewDim::at(MatmulOperand::Lhs, 3),
            channels_out: ViewDim::at(MatmulOperand::Out, 3),
            kernel: [
                ViewDim::at(MatmulOperand::Out, 0),
                ViewDim::at(MatmulOperand::Out, 1),
            ],
            out: [
                ViewDim::at(MatmulOperand::Rhs, 1),
                ViewDim::at(MatmulOperand::Rhs, 2),
            ],
        },
        extent_mismatch: ConvDeclined::WeightGradExtent,
    };

    const DATA_BACKWARD: Self = Self {
        free: ConvFreeAxes {
            batch: B,
            across: OH,
            m: OW,
            n: CI,
        },
        contracted: ConvContractedAxes {
            outer: RH,
            beside: RW,
            k: CO,
        },
        space: [B, OH, OW, CI, RH, RW, CO],
        lhs: [B, OH, OW, RH, RW, CO],
        rhs: [RH, RW, CO, CI],
        rhs_dims: [1, 2, 0, 3],
        out_dims: [0, 1, 2, 3],
        inner: MatmulOperands {
            lhs: CO,
            rhs: CI,
            out: CI,
        },
        gather: ConvGather::Gradient,
        extents: ConvExtents {
            channels_in: ViewDim::at(MatmulOperand::Out, 3),
            channels_out: ViewDim::at(MatmulOperand::Lhs, 3),
            kernel: [
                ViewDim::at(MatmulOperand::Rhs, 0),
                ViewDim::at(MatmulOperand::Rhs, 1),
            ],
            out: [
                ViewDim::at(MatmulOperand::Out, 1),
                ViewDim::at(MatmulOperand::Out, 2),
            ],
        },
        extent_mismatch: ConvDeclined::DataGradExtent,
    };

    /// The roles `operation` gives the axes of the space.
    ///
    /// # Errors
    ///
    /// [`ConvDeclined::Transposed`], for a transposed forward.
    pub fn of(operation: ConvolutionOperation) -> Result<Self, Refusal> {
        match operation {
            ConvolutionOperation::Forward => Ok(Self::FORWARD),
            ConvolutionOperation::BackwardWeight => Ok(Self::WEIGHT_BACKWARD),
            ConvolutionOperation::BackwardData => Ok(Self::DATA_BACKWARD),
            ConvolutionOperation::ForwardTransposed => Err(ConvDeclined::Transposed.into()),
        }
    }

    /// The spatial `[out, tap]` axis pairs, one per spatial dim of the gathered input.
    pub const SPATIAL: [[Axis; 2]; 2] = [[OH, RH], [OW, RW]];

    /// The axis a kernel leaves to its launch: the batch, which no vectorization or overhang
    /// decision reads. A batch that is a gathered tap axis (the weight backward's) stays compiled
    /// in, since the tile DSL's gather reads the extents of the axes it gathers over at expansion.
    pub fn dynamic(&self) -> Option<Axis> {
        let gathered = Self::SPATIAL.as_flattened().contains(&self.free.batch);
        (!gathered).then_some(self.free.batch)
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
    pub fn views(&self, shapes: MatmulOperands<[usize; 4]>) -> MatmulOperands<[usize; 4]> {
        MatmulOperands {
            lhs: shapes.lhs,
            rhs: self.rhs_dims.map(|dim| shapes.rhs[dim]),
            out: self.out_dims.map(|dim| shapes.out[dim]),
        }
    }

    /// The alignments of dense buffers of `shapes`, laid out as the operation's tensors are, once
    /// the rhs and the output are viewed through the roles ([`Self::rhs_view`],
    /// [`Self::out_view`]).
    pub fn dense_alignments(
        &self,
        shapes: MatmulOperands<[usize; 4]>,
    ) -> MatmulOperands<ElemAlignment> {
        let dense = |shape: [usize; 4], dims: [usize; 4]| {
            let strides = [
                shape[1] * shape[2] * shape[3],
                shape[2] * shape[3],
                shape[3],
                1,
            ];
            ElemAlignment::new(&dims.map(|dim| shape[dim]), &dims.map(|dim| strides[dim]))
        };
        MatmulOperands {
            lhs: dense(shapes.lhs, [0, 1, 2, 3]),
            rhs: dense(shapes.rhs, self.rhs_dims),
            out: dense(shapes.out, self.out_dims),
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
        let named = [(&self.rhs[..], rhs), (&self.free.in_order()[..], out)];
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
    /// the batch.
    pub fn plane_axes(&self) -> [Axis; 4] {
        [self.free.m, self.free.n, self.free.across, self.free.batch]
    }

    /// `rhs` with its dims listed in the order of [`Self::rhs`].
    ///
    /// # Errors
    ///
    /// [`ConvDeclined::PermutedTiled`], where the dims move and `rhs` is tiled.
    pub fn rhs_view(&self, rhs: TensorBinding) -> Result<TensorBinding, Refusal> {
        permuted(rhs, self.rhs_dims)
    }

    /// `out`, a tensor of the operation's output in its own layout, with its dims listed in the
    /// order of [`Self::free`].
    ///
    /// # Errors
    ///
    /// [`ConvDeclined::PermutedTiled`], where the dims move and `out` is tiled.
    pub fn out_view(&self, out: TensorBinding) -> Result<TensorBinding, Refusal> {
        permuted(out, self.out_dims)
    }
}

fn permuted(mut tensor: TensorBinding, dims: [usize; 4]) -> Result<TensorBinding, Refusal> {
    if dims == [0, 1, 2, 3] {
        return Ok(tensor);
    }
    if tensor.tiling.is_tiled() {
        return Err(ConvDeclined::PermutedTiled.into());
    }
    tensor.shape = Shape::from(dims.map(|dim| tensor.shape[dim]).to_vec());
    tensor.strides = Strides::new(&dims.map(|dim| tensor.strides[dim]));
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A transposed forward is no operation of the tiled convolution, which states it as the data
    /// backward at a stride.
    #[test]
    fn the_transposed_forward_is_refused_by_name() {
        assert_eq!(
            ConvRoles::of(ConvolutionOperation::ForwardTransposed),
            Err(ConvDeclined::Transposed.into())
        );
    }
}
