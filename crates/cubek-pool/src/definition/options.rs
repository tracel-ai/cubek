#[derive(Clone, Debug)]
pub struct MaxPoolOptions<const N: usize> {
    pub window: PoolWindow<N>,
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
            window: PoolWindow {
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
    pub window: PoolWindow<N>,
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
            window: PoolWindow {
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
pub struct PoolWindow<const N: usize> {
    pub kernel_size: [usize; N],
    pub stride: [usize; N],
    pub padding: [usize; N],
    pub ceil_mode: bool,
}
