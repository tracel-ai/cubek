use cubecl::{
    TestRuntime,
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceDtypes, ReducePrecision, ReduceStrategy, components::instructions::ReduceOperationConfig,
    reduce,
};
use cubek_test_utils::{
    Distribution, ExecutionOutcome, HostData, HostDataType, HostDataVec, StridedLayout, TestInput,
    TestOutcome, assert_equals_approx, launch_and_capture_outcome,
};

use cubek_reduce::eval::cpu_reference::{
    contiguous_strides, reference_all, reference_any, reference_argmax, reference_argmin,
    reference_argtopk, reference_max, reference_max_abs, reference_mean, reference_min,
    reference_prod, reference_sum, reference_topk,
};

pub struct TestCase {
    pub shape: Shape,
    pub stride: Strides,
    pub axis: Option<usize>,
    pub strategy: ReduceStrategy,
    pub input_dtype: StorageType,
    pub accumulation_dtype: StorageType,
}

impl core::fmt::Debug for TestCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestCase")
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .field("axis", &self.axis)
            .field("strategy", &self.strategy)
            .field("input_dtype", &self.input_dtype)
            .field("accumulation_dtype", &self.accumulation_dtype)
            .finish()
    }
}

impl TestCase {
    pub fn new<P: ReducePrecision>(
        shape: Shape,
        stride: Strides,
        axis: Option<usize>,
        strategy: ReduceStrategy,
    ) -> Self
    where
        P::EI: CubePrimitive,
        P::EA: CubePrimitive,
    {
        Self {
            shape,
            stride,
            axis,
            strategy,
            input_dtype: <P::EI as CubePrimitive>::as_type_native_unchecked().storage_type(),
            accumulation_dtype: <P::EA as CubePrimitive>::as_type_native_unchecked().storage_type(),
        }
    }

    pub fn test_sum(&self) {
        self.run_reduce_test(
            |input, axis| reference_sum(input, axis, None),
            self.input_dtype,
            ReduceOperationConfig::Sum,
            0.0625,
        );
    }

    pub fn test_mean(&self) {
        self.run_reduce_test(
            |input, axis| reference_mean(input, axis, None),
            self.input_dtype,
            ReduceOperationConfig::Mean,
            0.0625,
        );
    }

    pub fn test_prod(&self) {
        self.run_reduce_test(
            |input, axis| reference_prod(input, axis, None),
            self.input_dtype,
            ReduceOperationConfig::Prod,
            // Prod accumulates exponential error; use a loose relative bound.
            1.0,
        );
    }

    pub fn test_min(&self) {
        self.run_reduce_test(
            |input, axis| reference_min(input, axis, None),
            self.input_dtype,
            ReduceOperationConfig::Min,
            0.0625,
        );
    }

    pub fn test_max(&self) {
        self.run_reduce_test(
            |input, axis| reference_max(input, axis, None),
            self.input_dtype,
            ReduceOperationConfig::Max,
            0.0625,
        );
    }

    pub fn test_max_abs(&self) {
        self.run_reduce_test(
            |input, axis| reference_max_abs(input, axis, None),
            self.input_dtype,
            ReduceOperationConfig::MaxAbs,
            0.0625,
        );
    }

    pub fn test_any(&self) {
        // Bernoulli with p ≈ 1.5/axis_len makes ~22% of reduced slices all-zero
        // (any = 0) and the rest contain a non-zero (any = 1), so both merge
        // outcomes are exercised instead of the trivial all-ones case `uniform`
        // would produce. The output is u32 like the bool tensor backing callers
        // request (u32 is the one flag storage every test runtime supports), so
        // the in-kernel flag conversion is covered for every input dtype of the
        // matrix.
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test_with(
            |input, axis| reference_any(input, axis, None),
            u32_dtype,
            ReduceOperationConfig::Any,
            0.0,
            Distribution::Bernoulli(self.flag_probability()),
        );
    }

    pub fn test_all(&self) {
        // Mirror of `test_any`: p ≈ 1 - 1.5/axis_len makes ~22% of slices
        // all-ones (all = 1) and the rest contain a zero (all = 0).
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test_with(
            |input, axis| reference_all(input, axis, None),
            u32_dtype,
            ReduceOperationConfig::All,
            0.0,
            Distribution::Bernoulli(1.0 - self.flag_probability()),
        );
    }

    /// Per-element probability of `1` that yields a genuine mix of 0/1 reduction
    /// outputs for the current reduce axis length.
    fn flag_probability(&self) -> f32 {
        let axis_len = self.shape[self.axis.unwrap()].max(1) as f32;
        (1.5 / axis_len).clamp(0.0, 1.0)
    }

    pub fn test_argmax(&self) {
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test(
            |input, axis| reference_argmax(input, axis, None),
            u32_dtype,
            ReduceOperationConfig::ArgMax,
            0.0,
        );
    }

    pub fn test_argmin(&self) {
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test(
            |input, axis| reference_argmin(input, axis, None),
            u32_dtype,
            ReduceOperationConfig::ArgMin,
            0.0,
        );
    }

    pub fn test_argtopk(&self, k: usize) {
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test(
            move |input, axis| reference_argtopk(input, axis, k, None),
            u32_dtype,
            ReduceOperationConfig::ArgTopK(k),
            0.0,
        );
    }

    pub fn test_topk(&self, k: usize) {
        self.run_reduce_test(
            move |input, axis| reference_topk(input, axis, k, None),
            self.input_dtype,
            ReduceOperationConfig::TopK(k),
            1e-7,
        );
    }

    fn run_reduce_test(
        &self,
        reference: impl FnOnce(&HostData, usize) -> HostData,
        output_dtype: StorageType,
        config: ReduceOperationConfig,
        epsilon: f32,
    ) {
        self.run_reduce_test_with(
            reference,
            output_dtype,
            config,
            epsilon,
            Distribution::Uniform(-1., 1.),
        );
    }

    fn run_reduce_test_with(
        &self,
        reference: impl FnOnce(&HostData, usize) -> HostData,
        output_dtype: StorageType,
        config: ReduceOperationConfig,
        epsilon: f32,
        distribution: Distribution,
    ) {
        let client = TestRuntime::client(&Default::default());
        let axis = self.axis.unwrap();

        let (input_handle, input_host) = TestInput::builder(client.clone(), self.shape.clone())
            .dtype(self.input_dtype)
            .layout(StridedLayout::Explicit(
                self.stride.iter().copied().collect(),
            ))
            .random(1234, distribution)
            .generate_with_f32_host_data();

        let expected = cast_host_through_dtype(reference(&input_host, axis), output_dtype);

        let output_handle =
            self.build_output_tensor(&client, output_dtype, &expected.shape, &config);

        let strategy = self.strategy.clone();
        let dtypes = ReduceDtypes {
            input: self.input_dtype,
            output: output_dtype,
            accumulation: self.accumulation_dtype,
        };
        let input_binding = input_handle.binding();
        let output_binding = output_handle.clone().binding();
        let outcome = launch_and_capture_outcome(&client, |c| {
            reduce::<TestRuntime>(
                c,
                input_binding,
                output_binding,
                axis,
                strategy,
                config,
                dtypes,
            )
            .into()
        });

        let outcome = match outcome {
            ExecutionOutcome::Executed => {
                let actual =
                    HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
                assert_equals_approx(&actual, &expected, epsilon).as_test_outcome()
            }
            ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
        };
        outcome.enforce();
    }

    fn build_output_tensor(
        &self,
        client: &cubecl::client::ComputeClient<TestRuntime>,
        output_dtype: StorageType,
        output_shape: &Shape,
        config: &ReduceOperationConfig,
    ) -> TensorHandle<TestRuntime> {
        let axis = self.axis.unwrap();
        let is_parallel = self.stride[axis] == 1;
        let strides = match config {
            ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) if is_parallel => {
                parallel_multiple_output_strides(self.shape.as_slice(), &self.stride, axis, *k)
            }
            _ => contiguous_strides(output_shape.as_slice()),
        };
        TestInput::builder(client.clone(), output_shape.clone())
            .dtype(output_dtype)
            .layout(StridedLayout::Explicit(strides.iter().copied().collect()))
            .zeros()
            .generate_without_host_data()
    }
}

/// Output strides for multiple accumulators (like argtopk) in parallel mode.
fn parallel_multiple_output_strides(
    input_shape: &[usize],
    input_strides: &[usize],
    reduce_axis: usize,
    k: usize,
) -> Strides {
    let rank = input_shape.len();
    let size_r = input_shape[reduce_axis];

    let v = (0..rank)
        .filter(|&d| d != reduce_axis)
        .min_by_key(|&d| input_strides[d])
        .unwrap_or(0);
    let size_v = input_shape[v];

    let mut out = vec![0usize; rank];
    out[reduce_axis] = size_v;
    out[v] = 1;
    for d in 0..rank {
        if d == reduce_axis || d == v {
            continue;
        }
        out[d] = input_strides[d] * k / size_r;
    }
    Strides::new(&out)
}

/// Cast expected values through the GPU output dtype so comparisons account for
/// the precision loss that occurs when the kernel stores to a narrower type
/// (e.g. an f32 accumulator overflows once written to an f16 output).
fn cast_host_through_dtype(mut host: HostData, dtype: StorageType) -> HostData {
    if let HostDataVec::F32(values) = &host.data {
        let casted = match dtype {
            StorageType::Scalar(ElemType::Float(FloatKind::F16)) => values
                .iter()
                .map(|&x| half::f16::from_f32(x).to_f32())
                .collect(),
            StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => values
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_f32())
                .collect(),
            _ => return host,
        };
        host.data = HostDataVec::F32(casted);
    }
    host
}
