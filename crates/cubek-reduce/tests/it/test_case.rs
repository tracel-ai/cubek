use cubecl::{
    ir::{ElemType, FloatKind, StorageType},
    prelude::*,
    server::LaunchError,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
    {TestRuntime, server::ServerError},
};
use cubek_reduce::{
    ReduceDtypes, ReducePrecision, ReduceStrategy, components::instructions::ReduceOperationConfig,
    reduce,
};
use cubek_test_utils::{
    DataKind, ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput,
    TestOutcome, assert_equals_approx,
};

use crate::it::reference::{
    contiguous_strides, reference_argmax, reference_argmin, reference_argtopk, reference_max,
    reference_max_abs, reference_mean, reference_min, reference_prod, reference_sum,
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
            1.0,
        );
    }

    pub fn test_min(&self) {
        self.run_reduce_test(
            |input, axis| reference_min(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Min,
            0.0625,
        );
    }

    pub fn test_max(&self) {
        self.run_reduce_test(
            |input, axis| reference_max(input, axis),
            self.input_dtype,
            ReduceOperationConfig::Max,
            0.0625,
        );
    }

    pub fn test_max_abs(&self) {
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

        let outcome = match ExecutionOutcome::from(result) {
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
    ) -> TensorHandle<TestRuntime> {
        let strides = contiguous_strides(output_shape.as_slice());
        TestInput::new(
            client.clone(),
            output_shape.clone(),
            output_dtype,
            StrideSpec::Custom(strides.iter().copied().collect()),
            DataKind::Zeros,
        )
        .generate()
    }

    /// Build the device tensor and a matching logical-layout host reference.
    ///
    /// The host reference uses contiguous strides over the original shape so
    /// reference functions can iterate with plain logical coordinates. Broadcast
    /// inputs (stride == 0) are safe because [`logical_input_data`] produces the
    /// same value for every logical coordinate that maps to the same physical
    /// offset.
    fn setup_input(
        &self,
        client: &cubecl::client::ComputeClient<TestRuntime>,
    ) -> (TensorHandle<TestRuntime>, HostData) {
        let logical_data = self.logical_input_data();

        let tensor_handle = TestInput::new(
            client.clone(),
            self.shape.clone(),
            self.input_dtype,
            StrideSpec::Custom(self.stride.iter().copied().collect()),
            DataKind::Custom {
                data: logical_data.clone(),
            },
        )
        .generate();

        let host_values = round_trip_to_f32(&logical_data, self.input_dtype);
        let host = HostData {
            data: HostDataVec::F32(host_values),
            shape: self.shape.clone(),
            strides: contiguous_strides(self.shape.as_slice()),
        };

        (tensor_handle, host)
    }

    /// Deterministic values at each logical coordinate.
    ///
    /// Values are drawn from `{±0.125, ±0.25, …, ±1.0}` — all magnitudes ≤ 1,
    /// so the product of any subset is bounded and cannot overflow f32 under
    /// any reduction order (needed for `test_prod` on long axes).
    ///
    /// For broadcast dims (stride == 0) we zero the coord before hashing so
    /// every logical index mapping to the same physical offset yields the
    /// same value — required for the device-side scatter to be well defined.
    fn logical_input_data(&self) -> Vec<f32> {
        let shape = self.shape.as_slice();
        let rank = shape.len();
        let num_logical: usize = shape.iter().product();
        let mut data = Vec::with_capacity(num_logical);
        let mut coord = vec![0usize; rank];

        for linear in 0..num_logical {
            let mut rem = linear;
            for d in (0..rank).rev() {
                coord[d] = rem % shape[d];
                rem /= shape[d];
            }
            for d in 0..rank {
                if self.stride[d] == 0 {
                    coord[d] = 0;
                }
            }

            let mut hash: usize = 0;
            for (d, c) in coord.iter().enumerate() {
                hash = hash.wrapping_add(c.wrapping_mul(d.wrapping_add(31)));
            }
            let h = hash % 16;
            let magnitude = (h / 2 + 1) as f32 * 0.125;
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            data.push(sign * magnitude);
        }

        data
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
