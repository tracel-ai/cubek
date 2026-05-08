use cubecl::zspace::Shape;

#[derive(Clone, Debug)]
pub enum PoolProblem<const N: usize> {
    PoolForward(PoolForwardProblem<N>),
    PoolBackward(PoolBackwardProblem<N>),
}

#[derive(Clone, Debug)]
pub struct PoolForwardProblem<const N: usize> {
    pub input_shape: Shape,
    pub mode: PoolMode<N>,
}

#[derive(Clone, Debug)]
pub struct PoolBackwardProblem<const N: usize> {
    pub input_size: [usize; N],
    pub out_grad_shape: Shape,
    pub mode: PoolMode<N>,
}

#[derive(Clone, Debug)]
pub enum PoolMode<const N: usize> {
    Max(MaxPoolOptions<N>),
    Avg(AvgPoolOptions<N>),
    AdaptiveAvg(AdaptiveAvgPoolOptions<N>),
}

#[derive(Clone, Debug)]
pub struct MaxPoolOptions<const N: usize> {
    pub window: SlidingWindow<N>,
    pub dilation: [usize; N],
}

#[derive(Clone, Debug)]
pub struct AvgPoolOptions<const N: usize> {
    pub window: SlidingWindow<N>,
    pub count_include_pad: [bool; N],
}

#[derive(Clone, Debug)]
pub struct AdaptiveAvgPoolOptions<const N: usize> {
    pub output_size: [usize; N],
}

#[derive(Clone, Debug)]
pub struct SlidingWindow<const N: usize> {
    pub kernel_size: [usize; N],
    pub stride: [usize; N],
    pub padding: [usize; N],
    pub ceil_mode: [bool; N],
}

pub trait PoolGeometry<const N: usize> {
    fn output_shape(&self, input_shape: &[usize; N]) -> [usize; N];
}

impl<const N: usize> PoolGeometry<N> for MaxPoolOptions<N> {
    fn output_shape(&self, input_shape: &[usize; N]) -> [usize; N] {
        let mut out = [0; N];
        for i in 0..N {
            let effective_kernel = (self.window.kernel_size[i] - 1) * self.dilation[i] + 1;
            let padded = input_shape[i] + 2 * self.window.padding[i];

            let size = (padded - effective_kernel) as f32 / self.window.stride[i] as f32;
            out[i] = if self.window.ceil_mode[i] {
                f32::ceil(size) as usize + 1
            } else {
                f32::floor(size) as usize + 1
            };
        }
        out
    }
}

impl<const N: usize> PoolGeometry<N> for AvgPoolOptions<N> {
    fn output_shape(&self, input_shape: &[usize; N]) -> [usize; N] {
        let mut out = [0; N];
        for i in 0..N {
            let padded = input_shape[i] + 2 * self.window.padding[i];

            let size = (padded - self.window.kernel_size[i]) as f32 / self.window.stride[i] as f32;
            out[i] = if self.window.ceil_mode[i] {
                f32::ceil(size) as usize + 1
            } else {
                f32::floor(size) as usize + 1
            };
        }
        out
    }
}

impl<const N: usize> PoolGeometry<N> for AdaptiveAvgPoolOptions<N> {
    fn output_shape(&self, _input_shape: &[usize; N]) -> [usize; N] {
        self.output_size
    }
}
