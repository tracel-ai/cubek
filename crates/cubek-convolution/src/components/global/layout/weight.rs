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
    ConvolutionOperation, ConvolutionParams, ConvolutionProblem, global::layout::NhwcCoords,
};

/// Maps a 4D weight tensor of shape `(out_c, (k_h, k_w, in_c))` to a col-major 2D matmul tile with
/// shape `(n, k)`
#[derive(CubeType, CubeLaunch, Clone)]
pub struct WeightLayout {
    /// Number of channels, including padding, used for decomposing `k`
    pub padded_channels: FastDivmod<u32>,

    /// Shape of the combined kernel and channels dim, including padding
    pub rows: u32,
    /// Shape of the `out_c` dimension
    pub cols: u32,

    /// Size of the convolution kernel
    #[cube(comptime)]
    pub params: ConvolutionParams,
    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalLayoutConfig,
}

#[cube]
impl WeightLayout {
    pub fn new<E: Numeric, G: GlobalConfig>(
        rows: u32,
        cols: u32,
        padded_channels: FastDivmod<u32>,
        #[comptime] config: GlobalLayoutConfig,
        #[comptime] params: ConvolutionParams,
    ) -> WeightLayout {
        WeightLayout {
            rows,
            cols,
            padded_channels,
            config,
            params,
        }
    }
}

#[cube]
impl Layout for WeightLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, coords: Self::Coordinates) -> NhwcCoords {
        let params = self.params.comptime();
        let (_, k, n) = coords;

        let (mut rem, k_channel) = self.padded_channels.div_mod(k);

        let spatial_dims = params.dimensionality.num_dims();
        let mut kernel_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let dim = spatial_dims - i - 1;
            let ksize = params.kernel_size[dim];
            let k_pos = rem % ksize;
            rem /= ksize;

            kernel_pos.push(k_pos as i32);
        }

        let kernel_pos = kernel_pos.rev();

        let (batch, channel) = match params.operation {
            ConvolutionOperation::Forward | ConvolutionOperation::BackwardWeight => (n, k_channel),
            ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                (k_channel, n)
            }
        };

        NhwcCoords {
            batch,
            spatial: kernel_pos,
            channel,
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (NhwcCoords, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.cols)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, k, n) = pos;
        let check_k = self.config.check_row_bounds;
        let check_n = self.config.check_col_bounds;
        (!check_k || k < self.rows) && (!check_n || n < self.cols)
    }
}

impl<'a, R: Runtime> WeightLayoutLaunch<'a, R> {
    pub fn from_args(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        match problem.operation {
            ConvolutionOperation::Forward
            | ConvolutionOperation::ForwardTransposed
            | ConvolutionOperation::BackwardData => Self::from_args_rhs(client, problem, config),
            ConvolutionOperation::BackwardWeight => Self::from_args_out(client, problem, config),
        }
    }

    fn from_args_rhs(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        let padded_channels = problem.padded_channels as u32;
        let padded_channels = FastDivmodArgs::<u32>::new(client, padded_channels);
        let shape_k = ScalarArg::new(problem.k as u32);
        let shape_n = ScalarArg::new(problem.n as u32);

        let params = ConvolutionParams::from_problem(problem);

        WeightLayoutLaunch::new(padded_channels, shape_k, shape_n, params, config)
    }

    fn from_args_out(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        let padded_channels = problem.padded_channels as u32;
        let padded_channels = FastDivmodArgs::<u32>::new(client, padded_channels);
        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_n = ScalarArg::new(problem.n as u32);

        let params = ConvolutionParams::from_problem(problem);

        WeightLayoutLaunch::new(padded_channels, shape_n, shape_m, params, config)
    }
}
