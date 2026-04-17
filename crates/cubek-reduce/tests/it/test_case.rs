use cubecl::{
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    server::LaunchError,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
    {TestRuntime, server::ServerError},
};
use cubek_reduce::{
    ReduceDtypes, ReduceError, ReducePrecision, ReduceStrategy,
    components::instructions::ReduceOperationConfig, launch::RoutineStrategy, reduce,
};
use cubek_test_utils::{
    DataKind, HostData, HostDataType, HostDataVec, StrideSpec, TestInput, assert_equals_approx,
};

use crate::it::reference::{
    reference_argmax, reference_argmin, reference_argtopk, reference_max, reference_max_abs,
    reference_mean, reference_min, reference_prod, reference_sum,
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
        println!("Printing test: {self:?}");
        self.run_reduce_test(
            |input, axis| reference_sum(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Sum,
            0.0625,
        );
    }

    pub fn test_mean(&self) {
        self.run_reduce_test(
            |input, axis| reference_mean(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Mean,
            0.0625,
        );
    }

    pub fn test_prod(&self) {
        self.run_reduce_test(
            |input, axis| reference_prod(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Prod,
            // Prod accumulates exponential error; use a loose relative bound.
            // Exact zeros are avoided in the input data generator.
            1.0,
        );
    }

    pub fn test_min(&self) {
        println!("Printing test: {self:?}");
        self.run_reduce_test(
            |input, axis| reference_min(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Min,
            0.0625,
        );
    }

    pub fn test_max(&self) {
        println!("Printing test: {self:?}");
        self.run_reduce_test(
            |input, axis| reference_max(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Max,
            0.0625,
        );
    }

    pub fn test_max_abs(&self) {
        println!("Printing test: {self:?}");
        self.run_reduce_test(
            |input, axis| reference_max_abs(input, axis),
            self.input_dtype,
            ReduceOperationConfig::MaxAbs,
            0.0625,
        );
    }

    pub fn test_argmax(&self) {
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test(
            |input, axis| reference_argmax(input, axis),
            u32_dtype,
            ReduceOperationConfig::ArgMax,
            0.0,
        );
    }

    pub fn test_argmin(&self) {
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test(
            |input, axis| reference_argmin(input, axis),
            u32_dtype,
            ReduceOperationConfig::ArgMin,
            0.0,
        );
    }

    pub fn test_argtopk(&self, k: u32) {
        let u32_dtype = u32::as_type_native_unchecked().storage_type();
        self.run_reduce_test(
            move |input, axis| reference_argtopk(input, axis, k),
            u32_dtype,
            ReduceOperationConfig::ArgTopK(k),
            0.0,
        );
    }

    fn run_reduce_test(
        &self,
        reference: impl FnOnce(&HostData, usize) -> HostData,
        output_dtype: StorageType,
        config: ReduceOperationConfig,
        epsilon: f32,
    ) {
        let client = TestRuntime::client(&Default::default());

        if let RoutineStrategy::Cube(_blueprint) = &self.strategy.routine
            && client.properties().hardware.num_cpu_cores.is_some()
        {
            let test_full = std::env::var("CUBEK_TEST_FULL").unwrap_or_else(|_| "0".to_string());
            match test_full.as_str() {
                "1" | "true" => {}
                _ => {
                    println!(
                        "Skipping cube tests on CPU, because they are long to run and can stall the CI"
                    );
                    return;
                }
            }
        }

        let axis = self.axis.unwrap();

        let (input_handle, input_host) = self.setup_input(&client);

        let expected = cast_host_through_dtype(reference(&input_host, axis), output_dtype);

        let output_handle = self.build_output_tensor(&client, output_dtype, &expected.shape);

        let result = reduce::<TestRuntime>(
            &client,
            input_handle.binding(),
            output_handle.clone().binding(),
            axis,
            self.strategy.clone(),
            config,
            ReduceDtypes {
                input: self.input_dtype,
                output: output_dtype,
                accumulation: self.accumulation_dtype,
            },
        );

        match client.flush() {
            Ok(_) => {}
            Err(ServerError::ServerUnhealthy { errors, .. }) =>
            {
                #[allow(clippy::never_loop)]
                for error in errors.iter() {
                    match error {
                        cubecl::server::ServerError::Launch(LaunchError::TooManyResources(_)) => {}
                        _ => panic!("{errors:?}"),
                    }
                }
            }
            Err(err) => panic!("{err:?}"),
        }

        match result {
            Ok(_) => {}
            Err(e) => {
                let is_ok = matches!(e, ReduceError::PlanesUnavailable)
                    || matches!(e, ReduceError::ImprecisePlaneDim)
                    || matches!(e, ReduceError::Validation { .. });

                let test_mode = match is_ok {
                    true => std::env::var("CUBEK_TEST_MODE").unwrap_or_else(|_| "skip".to_string()),
                    false => "unexpected_error".to_string(),
                };

                match test_mode.as_str() {
                    "skip" => {}
                    "verbose" => println!("Skipping: {e:?}"),
                    mode => panic!("TestMode='{mode}', the test didn't run:\n {e:?}"),
                };

                return;
            }
        }

        let actual = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

        assert_equals_approx(&actual, &expected, epsilon)
            .as_test_outcome()
            .enforce();
    }

    fn build_output_tensor(
        &self,
        client: &cubecl::client::ComputeClient<TestRuntime>,
        output_dtype: StorageType,
        output_shape: &Shape,
    ) -> TensorHandle<TestRuntime> {
        let strides: Vec<usize> = contiguous_strides_vec(output_shape.as_slice());
        TestInput::new(
            client.clone(),
            output_shape.clone(),
            output_dtype,
            StrideSpec::Custom(strides),
            DataKind::Zeros,
        )
        .generate()
    }

    /// Build the input tensor + reference HostData.
    ///
    /// We bypass `TestInput` because it allocates a buffer of size
    /// `product(shape)` and assumes a contiguous write pattern. That fails for two
    /// cases we want to cover:
    ///   * jumpy strides (e.g. `stride=[512, 1], shape=[256, 256]`) where the
    ///     physical extent exceeds `product(shape)`;
    ///   * broadcast strides (stride == 0) where the `TestInput` custom-data kernel
    ///     would race on overlapping offsets.
    ///
    /// We compute a physical buffer size from the shape/stride pair and populate it
    /// directly so the GPU and the host reference read exactly the same values.
    fn setup_input(
        &self,
        client: &cubecl::client::ComputeClient<TestRuntime>,
    ) -> (TensorHandle<TestRuntime>, HostData) {
        let physical_size = physical_buffer_size(&self.shape, &self.stride);
        let data = self.physical_input_data(physical_size);

        let handle = create_input_handle(client, self.input_dtype, &data);

        let tensor_handle = TensorHandle::new(
            handle,
            self.shape.clone(),
            self.stride.clone(),
            self.input_dtype,
        );

        // The host-side reference must see the exact f32 values the GPU will see,
        // which for non-f32 input dtypes means the data round-tripped through the
        // target dtype's precision.
        let host_data = round_trip_to_f32(&data, self.input_dtype);
        let host = HostData {
            data: HostDataVec::F32(host_data),
            shape: self.shape.clone(),
            strides: self.stride.clone(),
        };

        (tensor_handle, host)
    }

    /// Generate deterministic f32 values addressed by physical offset.
    ///
    /// For each physical offset we decode the logical coordinate (via the tensor
    /// strides) and hash it. Offsets that don't correspond to any logical index are
    /// filled with zero — they're never read by the reduce kernel via shape/stride
    /// indexing. The hash zeroes out any coordinate whose axis has stride 0 so
    /// broadcast reads return the same value for every broadcast index.
    ///
    /// Values are drawn from the 16 non-zero elements of
    /// `{-2, -1.75, …, -0.25, 0.25, …, 2}` to keep `prod` from collapsing to zero.
    fn physical_input_data(&self, physical_size: usize) -> Vec<f32> {
        let shape: &[usize] = self.shape.as_slice();
        let rank = shape.len();
        let stride: Vec<usize> = self.stride.iter().copied().collect();
        let num_logical: usize = shape.iter().product();

        let mut data = vec![0.0f32; physical_size];
        let mut coord = vec![0usize; rank];

        for linear in 0..num_logical {
            let mut rem = linear;
            for d in (0..rank).rev() {
                coord[d] = rem % shape[d];
                rem /= shape[d];
            }

            let mut offset = 0usize;
            for d in 0..rank {
                offset += coord[d] * stride[d];
            }

            let mut hash_coord = coord.clone();
            for d in 0..rank {
                if stride[d] == 0 {
                    hash_coord[d] = 0;
                }
            }
            let mut hash: usize = 0;
            for (d, c) in hash_coord.iter().enumerate() {
                hash = hash.wrapping_add(c.wrapping_mul(d.wrapping_add(31)));
            }
            let h = hash % 16;
            let magnitude = (h / 2 + 1) as f32 * 0.25;
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            data[offset] = sign * magnitude;
        }

        data
    }
}

fn physical_buffer_size(shape: &Shape, stride: &Strides) -> usize {
    let mut max_offset = 0usize;
    for (s, d) in stride.iter().zip(shape.iter()) {
        if *d > 0 && *s > 0 {
            max_offset += (d - 1) * s;
        }
    }
    max_offset + 1
}

fn create_input_handle(
    client: &cubecl::client::ComputeClient<TestRuntime>,
    dtype: StorageType,
    data: &[f32],
) -> cubecl::server::Handle {
    match dtype {
        StorageType::Scalar(ElemType::Float(FloatKind::F32)) => {
            client.create_from_slice(f32::as_bytes(data))
        }
        StorageType::Scalar(ElemType::Float(FloatKind::F16)) => {
            let casted: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
            client.create_from_slice(half::f16::as_bytes(&casted))
        }
        StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => {
            let casted: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
            client.create_from_slice(half::bf16::as_bytes(&casted))
        }
        other => panic!("Unsupported input dtype for reduce tests: {other:?}"),
    }
}

fn round_trip_to_f32(data: &[f32], dtype: StorageType) -> Vec<f32> {
    match dtype {
        StorageType::Scalar(ElemType::Float(FloatKind::F32)) => data.to_vec(),
        StorageType::Scalar(ElemType::Float(FloatKind::F16)) => data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect(),
        StorageType::Scalar(ElemType::Float(FloatKind::BF16)) => data
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_f32())
            .collect(),
        other => panic!("Unsupported input dtype for reduce tests: {other:?}"),
    }
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

fn contiguous_strides_vec(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut s = vec![0usize; n];
    s[n - 1] = 1;
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}
