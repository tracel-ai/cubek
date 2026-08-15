use crate::definition::{AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions};
use cubecl::zspace::Shape;

#[derive(Clone, Debug)]
pub enum PoolProblem {
    Forward(PoolForward),
    Backward(PoolBackward),
}

#[derive(Clone, Debug)]
pub enum PoolForward {
    D1(PoolForwardProblem<1>),
    D2(PoolForwardProblem<2>),
    D3(PoolForwardProblem<3>),
}

#[derive(Clone, Debug)]
pub enum PoolBackward {
    D1(PoolBackwardProblem<1>),
    D2(PoolBackwardProblem<2>),
    D3(PoolBackwardProblem<3>),
}

#[derive(Clone, Debug)]
pub struct PoolForwardProblem<const N: usize> {
    pub input_shape: Shape,
    pub with_indices: bool,
    pub mode: PoolMode<N>,
}

#[derive(Clone, Debug)]
pub struct PoolBackwardProblem<const N: usize> {
    pub input_size: [usize; N],
    pub out_grad_shape: Shape,
    pub with_indices: bool,
    pub mode: PoolMode<N>,
}

#[cfg(feature = "benchmarks")]
impl<const N: usize> PoolBackwardProblem<N> {
    /// Reconstruct the channels-last input shape from the stored problem dimensions.
    pub(crate) fn input_shape(&self) -> Shape {
        let mut shape = vec![self.out_grad_shape[0]];
        shape.extend_from_slice(&self.input_size);
        shape.push(self.out_grad_shape[N + 1]);
        Shape::from(shape)
    }
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
