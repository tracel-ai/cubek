use super::{B, CI, CO, OH, OW, RH, RW};
use cubecl::prelude::TensorBinding;
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, Strides};
use cubek_microkernel::matmul::{MatmulAxes, MatmulOperands};

/// What a cmma fragment load over a step of taps and channels is declined with.
pub const CMMA_STEP_ACROSS_TAPS: Refusal = Refusal::Declined {
    why: "a cmma fragment load over a step made of taps and channels faults with a misaligned address (CUDA 716), unresolved",
};

/// What a launch computes. Every operation contracts the same seven-axis space through the same
/// gathered input; only the roles of the axes move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvOperation {
    /// `out[b, oh, ow, co]` from the input and the weight, contracted over the taps and the
    /// channels.
    Forward,
    /// `weight_grad[co, rh, rw, ci]` from the input and the output gradient, contracted over the
    /// batch and the output positions.
    BackwardWeight,
    /// `in_grad[b, ih, iw, ci]` from the output gradient and the weight, contracted over the
    /// taps and the output channels: the gradient is read backward through the taps.
    BackwardData,
}

/// Which axis of the space plays which part in an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvRoles {
    /// The output's axes in the order the instruction reads them, `[batch, across, m, n]`: the
    /// batch and the one across take a tile each of a cube or a plane, after the rows and the
    /// columns.
    pub free: [Axis; 4],
    /// The contracted axes a box walks whole, `[x, beside, k]`. `beside` fills a step of the
    /// contraction where `k` alone is shorter than the instruction's `k`.
    pub contracted: [Axis; 3],
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
    /// The axis each operand's buffer is contiguous along: the one its lines run along.
    pub inner: MatmulOperands<Axis>,
}

impl ConvRoles {
    const FORWARD: Self = Self {
        free: [B, OH, OW, CO],
        contracted: [RH, RW, CI],
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
    };

    const WEIGHT_BACKWARD: Self = Self {
        free: [RH, RW, CI, CO],
        contracted: [B, OH, OW],
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
    };

    const DATA_BACKWARD: Self = Self {
        free: [B, OH, OW, CI],
        contracted: [RH, RW, CO],
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
    };

    /// The axes of the instruction's rows, columns and contracted cells.
    pub fn axes(&self) -> MatmulAxes {
        MatmulAxes {
            m: self.free[2],
            n: self.free[3],
            k: self.contracted[2],
        }
    }

    /// The second contracted axis.
    pub fn beside(&self) -> Axis {
        self.contracted[1]
    }

    /// The output axis a cube or a plane takes one tile of, after the rows and the columns.
    pub fn across(&self) -> Axis {
        self.free[1]
    }

    pub fn batch(&self) -> Axis {
        self.free[0]
    }

    /// `rhs` with its dims listed in the order of [`Self::rhs`].
    ///
    /// # Errors
    ///
    /// [`CONV_PERMUTED_TILED`], where the dims move and `rhs` is tiled.
    pub fn rhs_view(&self, rhs: TensorBinding) -> Result<TensorBinding, Refusal> {
        permuted(rhs, self.rhs_dims)
    }

    /// `out`, a tensor of the operation's output in its own layout, with its dims listed in the
    /// order of [`Self::free`].
    ///
    /// # Errors
    ///
    /// [`CONV_PERMUTED_TILED`], where the dims move and `out` is tiled.
    pub fn out_view(&self, out: TensorBinding) -> Result<TensorBinding, Refusal> {
        permuted(out, self.out_dims)
    }
}

/// What a launch declines before preparing anything: a tiled tensor whose dims the view reorders,
/// since the tiling labels the buffer's dims and would no longer match.
pub const CONV_PERMUTED_TILED: Refusal = Refusal::Declined {
    why: "the tiled convolution reorders a tensor's dims, which it cannot do to a tiled one",
};

fn permuted(mut tensor: TensorBinding, dims: [usize; 4]) -> Result<TensorBinding, Refusal> {
    if dims == [0, 1, 2, 3] {
        return Ok(tensor);
    }
    if tensor.tiling.is_tiled() {
        return Err(CONV_PERMUTED_TILED);
    }
    tensor.shape = Shape::from(dims.map(|dim| tensor.shape[dim]).to_vec());
    tensor.strides = Strides::new(&dims.map(|dim| tensor.strides[dim]));
    Ok(tensor)
}

impl ConvOperation {
    pub const fn roles(self) -> ConvRoles {
        match self {
            Self::Forward => ConvRoles::FORWARD,
            Self::BackwardWeight => ConvRoles::WEIGHT_BACKWARD,
            Self::BackwardData => ConvRoles::DATA_BACKWARD,
        }
    }

    /// What a cmma step made of two contracted axes is declined with, where it is: a weight
    /// backward walks output positions through cmma.
    pub fn cmma_step_declined(self) -> Option<Refusal> {
        match self {
            Self::Forward | Self::BackwardData => Some(CMMA_STEP_ACROSS_TAPS),
            Self::BackwardWeight => None,
        }
    }
}
