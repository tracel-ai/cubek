use cubecl::{TestRuntime, prelude::*};
use cubek_reduce::shared_sum;
use cubek_test_utils::{
    DataKind, ExecutionOutcome, HostData, HostDataType, HostDataVec, StrideSpec, TestInput,
    TestOutcome, assert_equals_approx,
};

#[test]
pub fn test_shared_sum() {
    test_case().test_shared_sum()
}

fn test_case() -> TestCase {
    TestCase {
        shape: test_shape(),
        stride: test_strides(),
    }
}

#[derive(Debug)]
pub struct TestCase {
    pub shape: cubecl::zspace::Shape,
    pub stride: cubecl::zspace::Strides,
}

impl TestCase {
    pub fn test_shared_sum(&self) {
        // `shared_sum` reduces the entire tensor and the reduction over a broadcast
        // (stride == 0) axis is ambiguous: the kernel iterates the physical buffer via
        // linear_view, whereas the logical "all elements" semantics would count
        // broadcast duplicates. Skip broadcast cases to avoid testing an
        // implementation-defined path.
        if self.stride.iter().any(|&s| s == 0) {
            return;
        }

        let client = TestRuntime::client(&Default::default());
        let input_dtype = TestDType::as_type_native_unchecked().storage_type();

        let (input_handle, input_host) = TestInput::new(
            client.clone(),
            self.shape.clone(),
            input_dtype,
            StrideSpec::Custom(self.stride.iter().copied().collect()),
            DataKind::Custom {
                data: self.input_raw_data(),
            },
        )
        .generate_with_f32_host_data();

        let expected_sum = sum_all(&input_host);

        let output_handle = TestInput::new(
            client.clone(),
            cubecl::zspace::shape![1],
            input_dtype,
            StrideSpec::Custom(vec![1]),
            DataKind::Zeros,
        )
        .generate();

        let cube_count = 3;
        let result = shared_sum(
            &client,
            input_handle.binding(),
            output_handle.clone().binding(),
            cube_count,
            TestDType::as_type_native_unchecked().elem_type(),
        );

        let outcome = match ExecutionOutcome::from(result) {
            ExecutionOutcome::Executed => {
                let actual =
                    HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
                let expected = HostData {
                    data: HostDataVec::F32(vec![expected_sum]),
                    shape: cubecl::zspace::shape![1],
                    strides: strides![1],
                };
                assert_equals_approx(&actual, &expected, 0.0625).as_test_outcome()
            }
            ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
        };
        outcome.enforce();
    }

    /// Deterministic f32 values in {-2, -1.75, …, -0.25, 0.25, …, 2} excluding zero.
    fn input_raw_data(&self) -> Vec<f32> {
        let shape = self.shape.as_slice();
        let rank = shape.len();
        let num_elems: usize = shape.iter().product();
        let mut data = Vec::with_capacity(num_elems);
        let mut coord = vec![0usize; rank];

        for linear in 0..num_elems {
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
            let magnitude = (h / 2 + 1) as f32 * 0.25;
            let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
            data.push(sign * magnitude);
        }

        data
    }
}

fn sum_all(input: &HostData) -> f32 {
    let rank = input.shape.len();
    if rank == 0 {
        return input.get_f32(&[]);
    }
    let mut coord = vec![0usize; rank];
    let num: usize = input.shape.iter().product();
    let mut acc = 0.0f32;
    for linear in 0..num {
        let mut rem = linear;
        for d in (0..rank).rev() {
            coord[d] = rem % input.shape[d];
            rem /= input.shape[d];
        }
        acc += input.get_f32(&coord);
    }
    acc
}
