mod adaptive_avg_pool2d;
mod adaptive_avg_pool3d;
mod avg_pool2d;
mod max_pool2d;

use cubecl::zspace::Shape;
use cubek_pool::definition::{PoolForwardProblem, PoolMode};

pub fn make_problem(
    input_shape: Shape,
    with_indices: bool,
    mode: impl Into<PoolMode<2>>,
) -> PoolForwardProblem<2> {
    PoolForwardProblem {
        input_shape,
        with_indices,
        mode: mode.into(),
    }
}
