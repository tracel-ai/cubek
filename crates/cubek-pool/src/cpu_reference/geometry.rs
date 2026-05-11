use crate::definition::{
    AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions, PoolForwardProblem, PoolMode,
};
use cubecl::zspace::Shape;

pub trait PoolGeometry<const N: usize> {
    fn output_shape(&self, input_shape: &Shape) -> Shape;
}

impl<const N: usize> PoolGeometry<N> for PoolForwardProblem<N> {
    fn output_shape(&self, input_shape: &Shape) -> Shape {
        self.mode.output_shape(input_shape)
    }
}

impl<const N: usize> PoolGeometry<N> for PoolMode<N> {
    fn output_shape(&self, input_shape: &Shape) -> Shape {
        match self {
            PoolMode::Max(opts) => opts.output_shape(input_shape),
            PoolMode::Avg(opts) => opts.output_shape(input_shape),
            PoolMode::AdaptiveAvg(opts) => opts.output_shape(input_shape),
        }
    }
}

impl<const N: usize> PoolGeometry<N> for MaxPoolOptions<N> {
    fn output_shape(&self, input_shape: &Shape) -> Shape {
        let input_dims = input_shape.to_vec();
        let mut out = vec![input_dims[0]];
        for i in 0..N {
            let idx = 1 + i;
            let effective_kernel = (self.window.kernel_size[i] - 1) * self.dilation[i] + 1;
            let padded = input_dims[idx] + 2 * self.window.padding[i];
            out.push(pooled_dim(
                padded,
                effective_kernel,
                self.window.stride[i],
                self.window.ceil_mode,
            ));
        }
        out.push(input_dims[input_dims.len() - 1]);
        Shape::from(out)
    }
}

impl<const N: usize> PoolGeometry<N> for AvgPoolOptions<N> {
    fn output_shape(&self, input_shape: &Shape) -> Shape {
        let input_dims = input_shape.to_vec();
        let mut out = vec![input_dims[0]];
        for i in 0..N {
            let idx = 1 + i;
            let padded = input_dims[idx] + 2 * self.window.padding[i];
            let effective_kernel = self.window.kernel_size[i];
            out.push(pooled_dim(
                padded,
                effective_kernel,
                self.window.stride[i],
                self.window.ceil_mode,
            ));
        }
        out.push(input_dims[input_dims.len() - 1]);
        Shape::from(out)
    }
}

impl<const N: usize> PoolGeometry<N> for AdaptiveAvgPoolOptions<N> {
    fn output_shape(&self, _input_shape: &Shape) -> Shape {
        let input_dims = _input_shape.to_vec();
        let mut out = vec![input_dims[0]];
        out.extend_from_slice(&self.output_size);
        out.push(input_dims[input_dims.len() - 1]);
        Shape::from(out)
    }
}

fn pooled_dim(padded: usize, effective_kernel: usize, stride: usize, ceil_mode: bool) -> usize {
    let size = (padded as f32 - effective_kernel as f32) / stride as f32;
    (if ceil_mode { size.ceil() } else { size.floor() }) as usize + 1
}
