use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::definition::{PoolBackward, PoolForward, PoolProblem};
use crate::eval::benchmarks::strategy::PoolStrategy;
use crate::{pool2d, pool2d_backward, pool2d_with_indices, pool2d_with_indices_backward};

pub fn bench(
    _strategy: &PoolStrategy,
    problem: &PoolProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let dtype = f32::as_type_native_unchecked().storage_type();

    let bench = PoolBench {
        problem: problem.clone(),
        device,
        client,
        dtype,
        indices_dtype: i32::as_type_native_unchecked().storage_type(),
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct PoolBench {
    problem: PoolProblem,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    dtype: StorageType,
    indices_dtype: StorageType,
    samples: usize,
}

#[derive(Clone)]
enum PoolBenchInput {
    Forward(TensorHandle<TestRuntime>),
    Backward {
        input: TensorHandle<TestRuntime>,
        out_grad: TensorHandle<TestRuntime>,
        indices: Option<TensorHandle<TestRuntime>>,
    },
}

impl Benchmark for PoolBench {
    type Input = PoolBenchInput;
    type Output = TensorHandle<TestRuntime>;

    fn prepare(&self) -> Self::Input {
        match &self.problem {
            PoolProblem::Forward(PoolForward::D2(prob)) => {
                let input = TestInput::builder(self.client.clone(), prob.input_shape.clone())
                    .dtype(self.dtype)
                    .uniform(0, -1., 1.)
                    .generate_without_host_data();
                PoolBenchInput::Forward(input)
            }
            PoolProblem::Backward(PoolBackward::D2(prob)) => {
                let n = prob.out_grad_shape[0];
                let c = prob.out_grad_shape[3];
                let input_shape = Shape::from(vec![n, prob.input_size[0], prob.input_size[1], c]);

                let input = TestInput::builder(self.client.clone(), input_shape.clone())
                    .dtype(self.dtype)
                    .uniform(0, -1., 1.)
                    .generate_without_host_data();

                let out_grad = TestInput::builder(self.client.clone(), prob.out_grad_shape.clone())
                    .dtype(self.dtype)
                    .uniform(1, -1., 1.)
                    .generate_without_host_data();

                let indices = if prob.with_indices {
                    let output_shape = &input.shape();
                    let output =
                        TensorHandle::empty(&self.client, output_shape.to_vec(), self.dtype);
                    let indices = TensorHandle::empty(
                        &self.client,
                        output_shape.to_vec(),
                        self.indices_dtype,
                    );

                    pool2d_with_indices(
                        &self.client,
                        input.clone().binding(),
                        output.binding(),
                        indices.clone().binding(),
                        prob.mode.clone(),
                        self.dtype,
                    )
                    .expect("failed to create pool indices");

                    Some(indices)
                } else {
                    None
                };

                PoolBenchInput::Backward {
                    input,
                    out_grad,
                    indices,
                }
            }
            _ => panic!("benchmarks only support 2d pool problems"),
        }
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<TestRuntime>, String> {
        match (&self.problem, input) {
            (PoolProblem::Forward(PoolForward::D2(prob)), PoolBenchInput::Forward(input)) => {
                let output_shape = &input.shape();
                let output = TensorHandle::empty(&self.client, output_shape.to_vec(), self.dtype);

                if prob.with_indices {
                    let indices = TensorHandle::empty(
                        &self.client,
                        output_shape.to_vec(),
                        self.indices_dtype,
                    );
                    pool2d_with_indices(
                        &self.client,
                        input.binding(),
                        output.clone().binding(),
                        indices.binding(),
                        prob.mode.clone(),
                        self.dtype,
                    )
                    .map_err(|err| format!("{err}"))?;
                } else {
                    pool2d(
                        &self.client,
                        input.binding(),
                        output.clone().binding(),
                        prob.mode.clone(),
                        self.dtype,
                    )
                    .map_err(|err| format!("{err}"))?;
                }

                Ok(output)
            }
            (
                PoolProblem::Backward(PoolBackward::D2(prob)),
                PoolBenchInput::Backward {
                    input,
                    out_grad,
                    indices,
                },
            ) => {
                let n = prob.out_grad_shape[0];
                let c = prob.out_grad_shape[3];
                let input_grad_shape = vec![n, prob.input_size[0], prob.input_size[1], c];
                let output = TensorHandle::empty(&self.client, input_grad_shape, self.dtype);

                if let Some(indices) = indices {
                    pool2d_with_indices_backward(
                        &self.client,
                        input.binding(),
                        out_grad.binding(),
                        indices.binding(),
                        output.clone().binding(),
                        prob.mode.clone(),
                        self.dtype,
                        self.indices_dtype,
                    )
                    .map_err(|err| format!("{err}"))?;
                } else {
                    pool2d_backward(
                        &self.client,
                        input.binding(),
                        out_grad.binding(),
                        output.clone().binding(),
                        prob.mode.clone(),
                        self.dtype,
                    )
                    .map_err(|err| format!("{err}"))?;
                }

                Ok(output)
            }
            _ => Err("benchmarks only support 2d pool problems".to_string()),
        }
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        match &self.problem {
            PoolProblem::Forward(PoolForward::D2(prob)) => format!(
                "pool-{:?}-{:?}-{:?}-{:?}-indices-{:?}",
                self.dtype, prob.mode, self.device, prob.input_shape, prob.with_indices,
            )
            .to_lowercase(),
            PoolProblem::Backward(PoolBackward::D2(prob)) => format!(
                "pool-backward-{:?}-{:?}-{:?}-{:?}-indices-{:?}",
                self.dtype, prob.mode, self.device, prob.out_grad_shape, prob.with_indices,
            )
            .to_lowercase(),
            _ => panic!("benchmarks only support 2d pool problems"),
        }
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
