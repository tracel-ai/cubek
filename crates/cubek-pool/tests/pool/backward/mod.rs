mod adaptive_avg_pool2d;
mod adaptive_avg_pool3d;
mod avg_pool2d;
mod max_pool2d;

use cubecl::zspace::Shape;
use cubek_pool::definition::{PoolBackwardProblem, PoolMode};

pub fn make_problem(
    input_size: [usize; 2],
    out_grad_shape: Shape,
    with_indices: bool,
    mode: impl Into<PoolMode<2>>,
) -> PoolBackwardProblem<2> {
    PoolBackwardProblem {
        input_size,
        out_grad_shape,
        with_indices,
        mode: mode.into(),
    }
}
