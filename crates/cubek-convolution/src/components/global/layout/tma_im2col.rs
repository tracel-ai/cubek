use cubecl::{
    prelude::*,
    std::{
        FastDivmod, FastDivmodArgs,
        tensor::layout::{CoordsDyn, Layout, LayoutExpand},
    },
};
use cubek_matmul::launch::BatchedCoords;

use crate::components::{
    ConvolutionOperation, ConvolutionParams, ConvolutionProblem, global::layout::NhwcCoords,
};

/// Im2col layout, producing both the position and offset
#[derive(CubeType, CubeLaunch)]
pub struct TmaIm2colLayout {
    shape_out: Sequence<FastDivmod<u32>>,
    padded_channels: FastDivmod<u32>,
    rows: u32,
    cols: u32,
    #[cube(comptime)]
    params: ConvolutionParams,
    #[cube(comptime)]
    check_kernel: bool,
}

#[cube]
impl TmaIm2colLayout {
    pub fn new(
        shape_out: Sequence<FastDivmod<u32>>,
        padded_channels: FastDivmod<u32>,
        rows: u32,
        cols: u32,
        #[comptime] params: ConvolutionParams,
        #[comptime] check_kernel: bool,
    ) -> Self {
        TmaIm2colLayout {
            shape_out,
            padded_channels,
            params,
            check_kernel,
            rows,
            cols,
        }
    }
}

#[cube]
impl Layout for TmaIm2colLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = (NhwcCoords, CoordsDyn);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, m, k) = pos;
        let params = self.params.comptime();

        let (n_offs, spatial_offsets) = div_mod_seq(m, &self.shape_out);
        let spatial_dims = spatial_offsets.len();

        let mut in_offs = Sequence::<i32>::new();

        #[unroll]
        for dim in 0..spatial_dims {
            let stride = params.stride[dim] as i32;
            let pad = params.padding[dim];
            let out_pos = spatial_offsets[dim] as i32;
            let offs = match params.operation {
                ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => {
                    out_pos * stride - pad
                }
                ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                    let ksize = params.kernel_size[dim] as i32;
                    (out_pos + pad - ((ksize - 1) * params.dilation[dim] as i32)) / stride
                }
            };
            in_offs.push(offs);
        }

        let (mut k_idx, channel_start) = self.padded_channels.div_mod(k);

        let mut pos = NhwcCoords {
            batch: n_offs,
            spatial: in_offs,
            channel: channel_start,
        };

        let mut k_offs = Sequence::new();
        let k_rank = params.dimensionality.num_dims();

        #[unroll]
        for i in 0..k_rank {
            let dim = k_rank - i - 1;
            let k_size = params.kernel_size[dim];
            let k_pos = k_idx % k_size;

            let k_pos = match params.operation {
                ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => k_pos,
                ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                    // Since kernels are always positive, we need to subtract the bottom right
                    // corner (see position above), then add the inverted index to it.
                    k_size - k_pos - 1
                }
            };
            k_offs.push(k_pos * params.dilation[dim]);
            k_idx /= k_size;
        }

        if self.check_kernel.comptime() {
            // This is the largest index that's aligned to the channel count in all cases.
            // Alignment is 256, and that's the largest tile size possible with TMA.
            // Could alternatively solve this by only loading if in bounds, and adjusting the awaited
            // bytes by the in-bounds tiles but that's more complicated than just trying to load a very
            // large channel index and letting bounds checks handle it.
            let kernel_mask = (k_idx > 0) as u32 * 0x7FFFFF00u32;
            pos.channel = pos.channel.max(kernel_mask);
        }

        (pos, k_offs.rev())
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.cols)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

/// Decompose a linear index into local positions along each dimension in `shape`. Also returns the
/// left over remainder.
#[cube]
pub(crate) fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod<u32>>) -> (u32, Sequence<u32>) {
    let rank = shape.len().comptime();
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;
        let (rem, offs_local) = shape[dim].div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}

impl<'a, R: Runtime> TmaIm2colLayoutLaunch<'a, R> {
    pub fn from_args(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        check_kernel: bool,
    ) -> Self {
        let shape_out = problem
            .out_shape
            .iter()
            .map(|it| FastDivmodArgs::<u32>::new(client, *it as u32))
            .collect();

        let padded_channels = problem.padded_channels as u32;
        let padded_channels = FastDivmodArgs::<u32>::new(client, padded_channels);
        let params = ConvolutionParams::from_problem(problem);

        match problem.operation {
            ConvolutionOperation::Forward
            | ConvolutionOperation::ForwardTransposed
            | ConvolutionOperation::BackwardData => {
                Self::from_args_lhs(problem, shape_out, padded_channels, params, check_kernel)
            }
            ConvolutionOperation::BackwardWeight => {
                Self::from_args_rhs(problem, shape_out, padded_channels, params, check_kernel)
            }
        }
    }

    fn from_args_lhs(
        problem: &ConvolutionProblem,
        shape_out: SequenceArg<'a, R, FastDivmod<u32>>,
        padded_channels: FastDivmodArgs<u32>,
        params: ConvolutionParams,
        check_kernel: bool,
    ) -> Self {
        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_k = ScalarArg::new(problem.k as u32);

        TmaIm2colLayoutLaunch::new(
            shape_out,
            padded_channels,
            shape_m,
            shape_k,
            params,
            check_kernel,
        )
    }

    fn from_args_rhs(
        problem: &ConvolutionProblem,
        shape_out: SequenceArg<'a, R, FastDivmod<u32>>,
        padded_channels: FastDivmodArgs<u32>,
        params: ConvolutionParams,
        check_kernel: bool,
    ) -> Self {
        let shape_k = ScalarArg::new(problem.k as u32);
        let shape_n = ScalarArg::new(problem.n as u32);

        TmaIm2colLayoutLaunch::new(
            shape_out,
            padded_channels,
            shape_k,
            shape_n,
            params,
            check_kernel,
        )
    }
}
