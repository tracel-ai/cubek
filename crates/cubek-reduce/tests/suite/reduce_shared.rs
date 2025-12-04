use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubek_reduce::shared_sum;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

static PRECISION: i32 = 4;

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
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

impl TestCase {
    pub fn test_shared_sum(&self) {
        let input_values: Vec<TestDType> = self.random_input_values();
        let mut expected = TestDType::from_int(0);
        for v in input_values.iter() {
            expected += *v;
        }
        self.run_shared_sum_test(input_values, expected);
    }

    pub fn run_shared_sum_test(&self, input_values: Vec<TestDType>, expected: TestDType) {
        let client = TestRuntime::client(&Default::default());

        let input_handle = client.create_from_slice(TestDType::as_bytes(&input_values));
        let output_handle =
            client.create_from_slice(TestDType::as_bytes(&[TestDType::from_int(0)]));

        let input = unsafe {
            TensorHandleRef::from_raw_parts(
                &input_handle,
                &self.stride,
                &self.shape,
                size_of::<TestDType>(),
            )
        };
        let output = unsafe {
            TensorHandleRef::from_raw_parts(&output_handle, &[1], &[1], size_of::<TestDType>())
        };

        let cube_count = 3;
        let result = shared_sum(
            &client,
            input,
            output,
            cube_count,
            TestDType::as_type_native_unchecked().elem_type(),
        );

        if result.is_err() {
            return; // don't execute the test in that case since atomic adds are not supported.
        }
        let bytes = client.read_one(output_handle);
        let actual = TestDType::from_bytes(&bytes);
        assert_approx_equal(actual, &[expected]);
    }

    fn random_input_values<F: Float>(&self) -> Vec<F> {
        let size = self.input_size();
        let rng = StdRng::seed_from_u64(self.pseudo_random_seed());
        let distribution = Uniform::new_inclusive(-2 * PRECISION, 2 * PRECISION).unwrap();
        let factor = 1.0 / (PRECISION as f32);
        distribution
            .sample_iter(rng)
            .take(size)
            .map(|r| F::new(r as f32 * factor))
            .collect()
    }

    fn input_size(&self) -> usize {
        let (stride, shape) = self
            .stride
            .iter()
            .zip(self.shape.iter())
            .max_by_key(|(stride, _)| *stride)
            .unwrap();
        stride * shape
    }

    // We don't need a fancy crypto-secure seed as this is only for testing.
    fn pseudo_random_seed(&self) -> u64 {
        123456789
    }
}

pub fn assert_approx_equal<N: Numeric>(actual: &[N], expected: &[N]) {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a = a.to_f32().unwrap();
        let e = e.to_f32().unwrap();
        let diff = (a - e).abs();
        if e == 0.0 {
            assert!(
                diff < 1e-10,
                "Values are not approx equal: index={i} actual={a}, expected={e}, difference={diff}",
            );
        } else {
            let rel_diff = diff / e.abs();
            assert!(
                rel_diff < 0.0625,
                "Values are not approx equal: index={i} actual={a}, expected={e}"
            );
        }
    }
}
