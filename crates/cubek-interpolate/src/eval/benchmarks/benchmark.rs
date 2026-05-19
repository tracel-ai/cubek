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

use crate::{definition::InterpolateProblem, eval::benchmarks::strategy::InterpolateStrategy};
use crate::{interpolate, interpolate_backward};

pub fn bench(
    _strategy: &InterpolateStrategy,
    problem: &InterpolateProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let dtype = f32::as_type_native_unchecked().storage_type();

    let bench = InterpolateBench {
        problem: problem.clone(),
        device,
        client,
        dtype,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct InterpolateBench {
    problem: InterpolateProblem,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    dtype: StorageType,
    samples: usize,
}

impl Benchmark for InterpolateBench {
    type Input = TensorHandle<TestRuntime>;
    type Output = TensorHandle<TestRuntime>;

    fn prepare(&self) -> Self::Input {
        let shape = match &self.problem {
            InterpolateProblem::Forward(prob) => Shape::new(prob.input_shape),
            InterpolateProblem::Backward(prob) => Shape::new(prob.out_grad_shape),
        };
        TestInput::builder(self.client.clone(), shape)
            .dtype(self.dtype)
            .uniform(0, -1., 1.)
            .generate_without_host_data()
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<TestRuntime>, String> {
        match &self.problem {
            InterpolateProblem::Forward(prob) => {
                let [n, _, _, c] = prob.input_shape;
                let output_shape = vec![n, prob.output_size[0], prob.output_size[1], c];
                let output = TensorHandle::empty(&self.client, output_shape, self.dtype);

                interpolate(
                    &self.client,
                    input.binding(),
                    output.clone().binding(),
                    prob.options,
                    self.dtype,
                )
                .map_err(|err| format!("{err}"))?;

                Ok(output)
            }
            InterpolateProblem::Backward(prob) => {
                let [n, _, _, c] = prob.out_grad_shape;
                let input_grad_shape = vec![n, prob.input_size[0], prob.input_size[1], c];

                // Random input tensor for the backward pass. The actual values don't matter
                // for benchmarking, so we just fill it with random data.
                let backward_input =
                    TestInput::builder(self.client.clone(), input_grad_shape.clone())
                        .dtype(self.dtype)
                        .uniform(0, -1., 1.)
                        .generate_without_host_data();

                let output = TensorHandle::empty(&self.client, input_grad_shape, self.dtype);

                interpolate_backward(
                    &self.client,
                    backward_input.binding(),
                    input.clone().binding(),
                    output.clone().binding(),
                    prob.options,
                    self.dtype,
                )
                .map_err(|err| format!("{err}"))?;

                Ok(output)
            }
        }
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        match &self.problem {
            InterpolateProblem::Forward(prob) => format!(
                "interpolate-{:?}-{:?}-{:?}-{:?}",
                self.dtype, prob.options.mode, self.device, prob.input_shape,
            )
            .to_lowercase(),
            InterpolateProblem::Backward(prob) => format!(
                "interpolate-backward-{:?}-{:?}-{:?}-{:?}",
                self.dtype, prob.options.mode, self.device, prob.out_grad_shape,
            )
            .to_lowercase(),
        }
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
