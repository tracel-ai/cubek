use cubecl::{prelude::*, zspace::Shape};

use crate::test_tensor::{
    base::{DataKind, Distribution, InputDataType, TestInput},
    strides::LayoutSpec,
};

/// Fluent builder for [`TestInput`].
///
/// Use [`TestInput::builder`] to start one. The builder holds the shape,
/// dtype, and layout spec. Call a finalizer (`arange`, `eye`, `zeros`,
/// `uniform`, `bernoulli`, `random`, `custom`) to produce a [`TestInput`]
/// ready to generate a tensor handle, host data, or test tensor.
///
/// # Example
///
/// ```ignore
/// use cubek_test_utils::{TestInput, StridedLayout, Distribution};
///
/// let (handle, host) = TestInput::builder(client, [4, 4])
///     .layout(StridedLayout::ColMajor)
///     .uniform( 0, -1.0, 1.0)
///     .generate_with_f32_host_data();
/// ```
pub struct TestInputBuilder {
    client: Client,
    shape: Shape,
    dtype: Option<InputDataType>,
    layout: LayoutSpec,
}

impl TestInputBuilder {
    /// Override the dtype. Defaults to f32.
    pub fn dtype(mut self, dtype: impl Into<InputDataType>) -> Self {
        self.dtype = Some(dtype.into());
        self
    }

    /// Override the layout. Defaults to a row-major base stride layout.
    pub fn layout(mut self, layout: impl Into<LayoutSpec>) -> Self {
        self.layout = layout.into();
        self
    }

    /// `0, 1, 2, …` in row-major order.
    pub fn arange(self) -> TestInput {
        self.finalize(DataKind::Arange { scale: None })
    }

    /// `arange` with each value multiplied by `scale`.
    pub fn arange_scaled(self, scale: f32) -> TestInput {
        self.finalize(DataKind::Arange { scale: Some(scale) })
    }

    /// Identity matrix (1 on the diagonal, 0 elsewhere).
    pub fn eye(self) -> TestInput {
        self.finalize(DataKind::Eye)
    }

    /// All-zeros tensor.
    pub fn zeros(self) -> TestInput {
        self.finalize(DataKind::Zeros)
    }

    /// Random tensor with a custom [`Distribution`].
    pub fn random(self, seed: u64, distribution: Distribution) -> TestInput {
        self.finalize(DataKind::Random { seed, distribution })
    }

    /// Uniform random in `[lo, hi]`.
    pub fn uniform(self, seed: u64, lo: f32, hi: f32) -> TestInput {
        self.random(seed, Distribution::Uniform(lo, hi))
    }

    /// Bernoulli random with probability `p` of 1.
    pub fn bernoulli(self, seed: u64, p: f32) -> TestInput {
        self.random(seed, Distribution::Bernoulli(p))
    }

    /// Normal (Gaussian) random with the given `mean` and `std`.
    pub fn normal(self, seed: u64, mean: f32, std: f32) -> TestInput {
        self.random(seed, Distribution::Normal { mean, std })
    }

    /// Tensor populated from an explicit row-major `Vec<f32>`.
    pub fn custom(self, data: Vec<f32>) -> TestInput {
        self.finalize(DataKind::Custom { data })
    }

    /// Evenly-spaced values from `start` to `end` inclusive, populated in
    /// row-major order. The number of points equals the tensor's element count.
    ///
    /// Equivalent to NumPy's `np.linspace(start, end, num=shape.numel()).reshape(shape)`.
    pub fn linspace(self, start: f32, end: f32) -> TestInput {
        let num_elems: usize = self.shape.iter().product();
        let data = match num_elems {
            0 => Vec::new(),
            1 => vec![start],
            _ => evenly_spaced(start, end, num_elems),
        };
        self.finalize(DataKind::Custom { data })
    }
}

impl TestInputBuilder {
    /// An empty builder of `shape` on `client`, in f32 and row-major until told otherwise.
    pub(super) fn new(client: Client, shape: Shape) -> Self {
        Self {
            client,
            shape,
            dtype: None,
            layout: LayoutSpec::default(),
        }
    }

    fn finalize(self, data_kind: DataKind) -> TestInput {
        let dtype = self
            .dtype
            .unwrap_or_else(|| InputDataType::Standard(f32::elem_type_native()));
        TestInput::new(self.client, self.shape, dtype, self.layout, data_kind)
    }
}
/// `count` values from `start` to `end`, both included, evenly spaced.
fn evenly_spaced(start: f32, end: f32, count: usize) -> Vec<f32> {
    let step = (end - start) / (count - 1) as f32;
    (0..count).map(|i| start + step * i as f32).collect()
}
