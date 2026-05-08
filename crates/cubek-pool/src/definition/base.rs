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

impl<const N: usize> From<MaxPoolOptions<N>> for PoolMode<N> {
    fn from(options: MaxPoolOptions<N>) -> Self {
        PoolMode::Max(options)
    }
}

impl<const N: usize> From<AvgPoolOptions<N>> for PoolMode<N> {
    fn from(options: AvgPoolOptions<N>) -> Self {
        PoolMode::Avg(options)
    }
}

impl<const N: usize> From<AdaptiveAvgPoolOptions<N>> for PoolMode<N> {
    fn from(options: AdaptiveAvgPoolOptions<N>) -> Self {
        PoolMode::AdaptiveAvg(options)
    }
}

#[derive(Clone, Debug)]
pub struct MaxPoolOptions<const N: usize> {
    pub window: SlidingWindow<N>,
    pub dilation: [usize; N],
}

impl<const N: usize> MaxPoolOptions<N> {
    pub fn new(
        kernel_size: [usize; N],
        stride: [usize; N],
        padding: [usize; N],
        dilation: [usize; N],
        ceil_mode: bool,
    ) -> Self {
        Self {
            window: SlidingWindow {
                kernel_size,
                stride,
                padding,
                ceil_mode,
            },
            dilation,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AvgPoolOptions<const N: usize> {
    pub window: SlidingWindow<N>,
    pub count_include_pad: bool,
}

impl<const N: usize> AvgPoolOptions<N> {
    pub fn new(
        kernel_size: [usize; N],
        stride: [usize; N],
        padding: [usize; N],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Self {
        Self {
            window: SlidingWindow {
                kernel_size,
                stride,
                padding,
                ceil_mode,
            },
            count_include_pad,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveAvgPoolOptions<const N: usize> {
    pub output_size: [usize; N],
}

impl<const N: usize> AdaptiveAvgPoolOptions<N> {
    pub fn new(output_size: [usize; N]) -> Self {
        Self { output_size }
    }
}

#[derive(Clone, Debug)]
pub struct SlidingWindow<const N: usize> {
    pub kernel_size: [usize; N],
    pub stride: [usize; N],
    pub padding: [usize; N],
    pub ceil_mode: bool,
}
