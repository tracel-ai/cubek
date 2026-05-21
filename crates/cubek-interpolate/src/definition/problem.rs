use crate::definition::InterpolateOptions;

#[derive(Clone, Debug)]
pub enum InterpolateProblem {
    Forward(InterpolateForwardProblem),
    Backward(InterpolateBackwardProblem),
}

#[derive(Clone, Debug)]
pub struct InterpolateForwardProblem {
    pub input_shape: [usize; 4],
    pub output_size: [usize; 2],
    pub options: InterpolateOptions,
}

#[derive(Clone, Debug)]
pub struct InterpolateBackwardProblem {
    pub input_size: [usize; 2],
    pub out_grad_shape: [usize; 4],
    pub options: InterpolateOptions,
}

impl InterpolateProblem {
    pub fn options(&self) -> InterpolateOptions {
        match self {
            InterpolateProblem::Forward(problem) => problem.options.clone(),
            InterpolateProblem::Backward(problem) => problem.options.clone(),
        }
    }
}
