mod kernel;

use cubek_interpolate::definition::{InterpolateForwardProblem, InterpolateOptions};

pub fn make_problem(
    input_shape: [usize; 4],
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> InterpolateForwardProblem {
    InterpolateForwardProblem::from_input_output_shapes(&input_shape.into(), &output_size, options)
}
