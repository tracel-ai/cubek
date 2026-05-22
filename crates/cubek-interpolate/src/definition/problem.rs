use cubecl::zspace::Shape;

use crate::definition::InterpolateOptions;

#[derive(Clone, Debug)]
pub enum InterpolateProblem {
    Forward(InterpolateForwardProblem),
    Backward(InterpolateBackwardProblem),
}

#[derive(Clone, Debug)]
pub struct InterpolateForwardProblem {
    pub batch: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub channels: usize,

    pub output_height: usize,
    pub output_width: usize,

    pub options: InterpolateOptions,
}

impl InterpolateForwardProblem {
    pub fn from_input_output_shapes(
        input_shape: &Shape,
        output_size: &[usize; 2],
        options: InterpolateOptions,
    ) -> Self {
        Self {
            batch: input_shape[0],
            input_height: input_shape[1],
            input_width: input_shape[2],
            channels: input_shape[3],
            output_height: output_size[0],
            output_width: output_size[1],
            options,
        }
    }

    pub fn input_shape(&self) -> Shape {
        [
            self.batch,
            self.input_height,
            self.input_width,
            self.channels,
        ]
        .into()
    }

    pub fn output_size(&self) -> Shape {
        [self.output_height, self.output_width].into()
    }

    pub fn output_shape(&self) -> Shape {
        [
            self.batch,
            self.output_height,
            self.output_width,
            self.channels,
        ]
        .into()
    }
}

#[derive(Clone, Debug)]
pub struct InterpolateBackwardProblem {
    pub input_size: [usize; 2],
    pub out_grad_shape: [usize; 4],
    pub options: InterpolateOptions,
}
