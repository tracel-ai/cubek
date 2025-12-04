use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubek_reduce::{
    ReduceDtypes, ReduceError, ReducePrecision, ReduceStrategy, instructions::*, reduce,
};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

static PRECISION: i32 = 4;

#[test]
pub fn test_argmax() {
    test_case().test_argmax();
}

#[test]
pub fn test_argmin() {
    test_case().test_argmin();
}

#[test]
pub fn test_mean() {
    test_case().test_sum();
}

#[test]
pub fn test_sum() {
    test_case().test_mean();
}

#[test]
pub fn test_prod() {
    test_case().test_prod();
}

fn test_case() -> TestCase {
    TestCase {
        shape: test_shape(),
        stride: test_strides(),
        axis: test_axis(),
        strategy: Some(ReduceStrategy {
            use_planes: test_use_plane(),
            shared: test_shared(),
        }),
    }
}

type EI = <TestDType as ReducePrecision>::EI;

#[derive(Debug)]
pub struct TestCase {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub axis: Option<usize>,
    pub strategy: Option<ReduceStrategy>,
}

impl TestCase {
    pub fn test_argmax(&self) {
        let input_values: Vec<<TestDType as ReducePrecision>::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => vec![0; input_values.len()],
            _ => self.cpu_argmax(&input_values),
        };
        self.run_reduce_test::<u32, ArgMax>(input_values, expected_values)
    }

    fn cpu_argmax<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::min_value(), 0_u32); self.num_output_values()];
        for (input_index, &value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                let (best, _) = expected[output_index];
                if value > best {
                    let coordinate = self.to_input_coordinate(input_index).unwrap();
                    expected[output_index] = (value, coordinate[self.axis.unwrap()] as u32);
                }
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    pub fn test_argmin(&self) {
        let input_values: Vec<<TestDType as ReducePrecision>::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => vec![0; input_values.len()],
            _ => self.cpu_argmin(&input_values),
        };
        self.run_reduce_test::<u32, ArgMin>(input_values, expected_values)
    }

    fn cpu_argmin<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::max_value(), 0_u32); self.num_output_values()];
        for (input_index, &value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                let (best, _) = expected[output_index];
                if value < best {
                    let coordinate = self.to_input_coordinate(input_index).unwrap();
                    expected[output_index] = (value, coordinate[self.axis.unwrap()] as u32);
                }
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    pub fn test_mean(&self) {
        let input_values: Vec<EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => input_values.clone(),
            _ => self.cpu_mean(&input_values),
        };
        self.run_reduce_test::<EI, Mean>(input_values, expected_values)
    }

    fn cpu_mean<F: Float>(&self, values: &[F]) -> Vec<F> {
        self.cpu_sum(values)
            .into_iter()
            .map(|sum| sum / F::new(self.shape[self.axis.unwrap()] as f32))
            .collect()
    }

    pub fn test_prod(&self) {
        let input_values: Vec<EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => input_values
                .iter()
                .map(|v| Self::powf(*v, self.shape[axis]))
                .collect(),
            _ => self.cpu_prod(&input_values),
        };
        self.run_reduce_test::<EI, Prod>(input_values, expected_values)
    }

    fn powf<F: Float>(base: F, power: usize) -> F {
        let mut result = F::new(1.0);
        for _ in 0..power {
            result *= base;
        }
        result
    }

    fn cpu_prod<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(1.0); self.num_output_values()];

        for (input_index, value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                expected[output_index] *= *value;
            }
        }
        expected
    }

    pub fn test_sum(&self) {
        let input_values: Vec<EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => input_values
                .iter()
                .map(|v| *v * EI::from_int(self.shape[axis] as i64))
                .collect(),
            _ => self.cpu_sum(&input_values),
        };
        self.run_reduce_test::<EI, Sum>(input_values, expected_values)
    }

    fn cpu_sum<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(0.0); self.num_output_values()];

        for (input_index, value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                expected[output_index] += *value;
            }
        }
        expected
    }

    pub fn run_reduce_test<O, K>(
        &self,
        input_values: Vec<<TestDType as ReducePrecision>::EI>,
        expected_values: Vec<O>,
    ) where
        O: Numeric + CubeElement + std::fmt::Display,
        K: ReduceFamily<Config = ()>,
    {
        let client = TestRuntime::client(&Default::default());

        let input_handle = client.create_from_slice(
            <<TestDType as ReducePrecision>::EI as CubeElement>::as_bytes(&input_values),
        );

        // Zero initialize a tensor with the same shape as input
        // except for the `self.axis` axis where the shape is 1.
        let output_handle =
            client.create_from_slice(O::as_bytes(&vec![O::from_int(0); expected_values.len()]));
        let mut output_shape = self.shape.clone();
        output_shape[self.axis.unwrap()] = 1;
        let output_stride = self.output_stride();

        let input = unsafe {
            TensorHandleRef::from_raw_parts(
                &input_handle,
                &self.stride,
                &self.shape,
                size_of::<TestDType>(),
            )
        };
        let output = unsafe {
            TensorHandleRef::from_raw_parts(
                &output_handle,
                &output_stride,
                &output_shape,
                size_of::<O>(),
            )
        };

        let result = reduce::<TestRuntime, K>(
            &client,
            input,
            output,
            self.axis.unwrap(),
            self.strategy,
            (),
            ReduceDtypes {
                input: <TestDType as ReducePrecision>::EI::as_type_native_unchecked(),
                output: O::as_type_native_unchecked(),
                accumulation: <TestDType as ReducePrecision>::EA::as_type_native_unchecked(),
            },
        );
        if result.is_err_and(|e| {
            matches!(e, ReduceError::PlanesUnavailable)
                || matches!(e, ReduceError::ImprecisePlaneDim)
        }) {
            return; // We don't test in that case.
        }

        let bytes = client.read_one(output_handle);
        let output_values = O::from_bytes(&bytes);
        assert_approx_equal(output_values, &expected_values);
    }

    fn num_output_values(&self) -> usize {
        self.shape.iter().product::<usize>() / self.shape[self.axis.unwrap()]
    }

    fn to_output_index(&self, input_index: usize) -> Option<usize> {
        let mut coordinate = self.to_input_coordinate(input_index)?;
        coordinate[self.axis.unwrap()] = 0;
        Some(self.from_output_coordinate(coordinate))
    }

    fn to_input_coordinate(&self, index: usize) -> Option<Vec<usize>> {
        let coordinate = self
            .stride
            .iter()
            .zip(self.shape.iter())
            .map(|(stride, shape)| {
                if *stride > 0 {
                    (index / stride) % shape
                } else {
                    index % shape
                }
            })
            .collect::<Vec<usize>>();
        self.validate_input_index(index, &coordinate)
            .then_some(coordinate)
    }

    fn validate_input_index(&self, index: usize, coordinate: &[usize]) -> bool {
        coordinate
            .iter()
            .zip(self.stride.iter())
            .map(|(c, s)| c * s)
            .sum::<usize>()
            == index
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_output_coordinate(&self, coordinate: Vec<usize>) -> usize {
        coordinate
            .into_iter()
            .zip(self.output_stride().iter())
            .map(|(c, s)| c * s)
            .sum()
    }

    fn output_stride(&self) -> Vec<usize> {
        self.shape
            .iter()
            .enumerate()
            .scan(1, |stride, (axis, shape)| {
                if axis == self.axis.unwrap() {
                    Some(1)
                } else {
                    let current = Some(*stride);
                    *stride *= shape;
                    current
                }
            })
            .collect()
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
        // (0..size).map(|x| F::from_int(x as i64)).collect() TODO DELETE
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
