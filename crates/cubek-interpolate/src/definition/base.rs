use crate::definition::InterpolateOptions;

#[derive(Clone, Debug)]
/// Description of an interpolate problem to solve, regardless of actual data
pub struct InterpolateProblem {
    pub input_shape: [usize; 4],
    pub output_size: [usize; 2],
    pub options: InterpolateOptions,
}
