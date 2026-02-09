use cubecl::prelude::*;
use cubecl::std::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Layout, LayoutExpand},
};
use cubek_matmul::{components::global::memory::GlobalLayoutConfig, launch::BatchedCoords};

use crate::components::{
    ConvolutionOperation, ConvolutionProblem,
    global::layout::{NhwcCoords, cast_seq, div_mod_seq},
};

/// Maps a 4D NHWC out tensor of shape `((n, h, w), c)` to a col-major 2D matmul tile with
/// shape `(m, n)`
#[derive(CubeType, CubeLaunch, Clone)]
pub struct OutLayout {
    /// Shape of DHW
    pub shape_out: Sequence<FastDivmod<u32>>,

    /// Shape of the conceptual `m` size
    pub rows: u32,
    /// Shape of the conceptual `n`size, or channels
    pub cols: u32,

    /// Global memory config for the backing tensor
    #[cube(comptime)]
    pub config: GlobalLayoutConfig,
}

#[cube]
impl OutLayout {
    pub fn new(
        rows: u32,
        cols: u32,
        shape_out: Sequence<FastDivmod<u32>>,
        #[comptime] config: GlobalLayoutConfig,
    ) -> OutLayout {
        OutLayout {
            shape_out,
            rows,
            cols,
            config,
        }
    }
}

#[cube]
impl Layout for OutLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = NhwcCoords;

    fn to_source_pos(&self, coords: Self::Coordinates) -> NhwcCoords {
        let (_, view_m, view_n) = coords;
        let (batch, spatial) = div_mod_seq(view_m, &self.shape_out);

        NhwcCoords {
            batch,
            spatial: cast_seq(spatial),
            channel: view_n,
        }
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (NhwcCoords, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.cols)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (_, row, col) = pos;
        (!self.config.check_row_bounds || row < self.rows)
            && (!self.config.check_col_bounds || col < self.cols)
    }
}

impl<'a, R: Runtime> OutLayoutLaunch<'a, R> {
    pub fn from_args(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        match problem.operation {
            ConvolutionOperation::Forward => Self::from_args_fprop(client, problem, config),
            ConvolutionOperation::ForwardTransposed | ConvolutionOperation::BackwardData => {
                Self::from_args_dgrad(client, problem, config)
            }
            ConvolutionOperation::BackwardWeight => Self::from_args_wgrad(client, problem, config),
        }
    }

    fn from_args_fprop(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        let shape_out = problem
            .out_shape
            .iter()
            .map(|s| FastDivmodArgs::<u32>::new(client, *s as u32))
            .collect();
        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_n = ScalarArg::new(problem.n as u32);

        Self::new(shape_out, shape_m, shape_n, config)
    }

    fn from_args_dgrad(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        let shape = problem
            .in_shape
            .iter()
            .map(|s| FastDivmodArgs::<u32>::new(client, *s as u32))
            .collect();
        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_n = ScalarArg::new(problem.n as u32);

        Self::new(shape, shape_m, shape_n, config)
    }

    fn from_args_wgrad(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        config: GlobalLayoutConfig,
    ) -> Self {
        let shape_out = problem
            .out_shape
            .iter()
            .map(|s| FastDivmodArgs::<u32>::new(client, *s as u32))
            .collect();
        let shape_m = ScalarArg::new(problem.m as u32);
        let shape_k = ScalarArg::new(problem.k as u32);

        Self::new(shape_out, shape_k, shape_m, config)
    }
}
