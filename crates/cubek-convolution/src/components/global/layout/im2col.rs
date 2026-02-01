use cubecl::prelude::*;
use cubecl::std::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Layout, LayoutExpand},
};
use cubek_matmul::{
    components::global::{GlobalConfig, memory::GlobalLayoutConfig},
    launch::BatchedCoords,
};

use crate::components::{
    ConvolutionOperation, ConvolutionParams, ConvolutionProblem,
    global::layout::{NhwcCoords, div_mod_seq},
};

/// Maps a 4D NHWC tensor to a 2D column matrix using the im2col transformation
/// It first decomposes the `(m, k)` matrix into `((n, out_h, out_w), (k_h, k_w, c))`, then applies
/// the convolution parameters to calculate the position in the input tensor for that kernel element.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct Im2colLayout {
    /// Shape of output DHW
    pub shape_out: Sequence<FastDivmod<u32>>,
    /// Shape of channel, for decomposing k
    pub padded_channels: FastDivmod<u32>,

    /// Shape of the combined `m` dimension, including padding
    pub rows: u32,
    /// Shape of the combined `k` dimension, including padding
    pub cols: u32,

    /// Comptime parameters for the convolution
    #[cube(comptime)]
    pub params: ConvolutionParams,
    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalLayoutConfig,
}

#[cube]
impl Im2colLayout {
    pub fn new<G: GlobalConfig>(
        rows: u32,
        cols: u32,
        padded_channels: FastDivmod<u32>,
        shape_out: Sequence<FastDivmod<u32>>,
        #[comptime] config: GlobalLayoutConfig,
        #[comptime] params: ConvolutionParams,
    ) -> Im2colLayout {
        Im2colLayout {
            shape_out,
            padded_channels,
            rows,
            cols,
            params,
            config,
        }
    }
}

#[cube]
impl Layout for Im2colLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, pos: Self::Coordinates) -> NhwcCoords {
        let params = self.params.comptime();
        let (_, view_m, view_k) = pos;

        let (batch, out_offs) = div_mod_seq(view_m, &self.shape_out);

        let (mut rem, channel) = self.padded_channels.div_mod(view_k);

        let spatial_dims = params.dimensionality.num_dims();
        let mut in_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let dim = spatial_dims - i - 1;
            let ksize = params.kernel_size[dim];
            let k_pos = (rem % ksize) as i32;
            rem /= ksize;

            let out_pos = out_offs[dim];
            let stride = params.stride[dim] as i32;
            let dilate = params.dilation[dim] as i32;
            let pad = params.padding[dim];

            let pos = match params.operation {
                ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => {
                    (out_pos as i32 * stride + k_pos * dilate) - pad
                }
                ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                    (out_pos as i32 + pad - k_pos * dilate) / stride
                }
            };
            in_pos.push(pos);
        }

        let in_pos = in_pos.rev();

        NhwcCoords {
            batch,
            spatial: in_pos,
            channel,
        }
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.cols)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (NhwcCoords, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, view_m, view_k) = pos;
        // Shouldn't be relied on because it doesn't check spatial
        let m_in_bounds = !self.config.check_row_bounds || view_m < self.rows;
        let k_in_bounds = !self.config.check_col_bounds || view_k < self.cols;
        m_in_bounds && k_in_bounds
    }
}

impl<'a, R: Runtime> Im2colLayoutLaunch<'a, R> {
    pub fn from_args(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        params: ConvolutionParams,
        config: GlobalLayoutConfig,
    ) -> Self {
        match problem.operation {
            ConvolutionOperation::Forward => Self::from_args_fprop(client, problem, params, config),
            ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                Self::from_args_dgrad(client, problem, params, config)
            }
            ConvolutionOperation::BackwardWeight => {
                Self::from_args_wgrad(client, problem, params, config)
            }
        }
    }

    fn from_args_fprop(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        params: ConvolutionParams,
        config: GlobalLayoutConfig,
    ) -> Self {
        let shape_out = problem
            .out_shape
            .iter()
            .map(|s| FastDivmodArgs::<u32>::new(client, *s as u32))
            .collect();

        let padded_channels = problem.padded_channels as u32;
        let padded_channels = FastDivmodArgs::<u32>::new(client, padded_channels);

        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_k = ScalarArg::new(problem.k as u32);

        Im2colLayoutLaunch::new(shape_out, padded_channels, shape_m, shape_k, params, config)
    }

    fn from_args_dgrad(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        params: ConvolutionParams,
        config: GlobalLayoutConfig,
    ) -> Self {
        let shape = problem
            .in_shape
            .iter()
            .map(|s| FastDivmodArgs::<u32>::new(client, *s as u32))
            .collect();

        let padded_channels = problem.padded_channels as u32;
        let padded_channels = FastDivmodArgs::<u32>::new(client, padded_channels);

        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_k = ScalarArg::new(problem.k as u32);

        Im2colLayoutLaunch::new(shape, padded_channels, shape_m, shape_k, params, config)
    }

    fn from_args_wgrad(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        params: ConvolutionParams,
        config: GlobalLayoutConfig,
    ) -> Self {
        let shape_out = problem
            .out_shape
            .iter()
            .map(|s| FastDivmodArgs::<u32>::new(client, *s as u32))
            .collect();

        let padded_channels = problem.padded_channels as u32;
        let padded_channels = FastDivmodArgs::<u32>::new(client, padded_channels);

        let shape_k = ScalarArg::new(problem.k as u32);
        let shape_n = ScalarArg::new(problem.n as u32);

        Im2colLayoutLaunch::new(shape_out, padded_channels, shape_k, shape_n, params, config)
    }
}
