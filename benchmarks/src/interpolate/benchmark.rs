use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};

use crate::interpolate::strategy::InterpolateStrategy;
use crate::registry::RunSamples;

use cubek::{interpolate::definition::InterpolateProblem, random::random_uniform};

pub fn bench(
    _strategy: &InterpolateStrategy,
    problem: &InterpolateProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    bench_on::<cubecl::TestRuntime>(
        Default::default(),
        f32::as_type_native_unchecked().storage_type(),
        problem,
        num_samples,
    )
}

pub fn bench_on<R: Runtime>(
    device: R::Device,
    dtype: StorageType,
    problem: &InterpolateProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);

    let bench = InterpolateBench::<R> {
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

struct InterpolateBench<R: Runtime> {
    problem: InterpolateProblem,
    device: R::Device,
    client: ComputeClient<R>,
    dtype: StorageType,
    samples: usize,
}

impl<R: Runtime> Benchmark for InterpolateBench<R> {
    type Input = TensorHandle<R>;
    type Output = TensorHandle<R>;

    fn prepare(&self) -> Self::Input {
        match &self.problem {
            InterpolateProblem::Forward(prob) => {
                let tensor =
                    TensorHandle::empty(&self.client, prob.input_shape.to_vec(), self.dtype);

                random_uniform(&self.client, -1., 1., tensor.clone().binding(), self.dtype)
                    .expect("Failed to initialize random values for InterpolateBench");

                tensor
            }
            InterpolateProblem::Backward(prob) => {
                let tensor =
                    TensorHandle::empty(&self.client, prob.out_grad_shape.to_vec(), self.dtype);

                random_uniform(&self.client, -1., 1., tensor.clone().binding(), self.dtype)
                    .expect("Failed to initialize random values for InterpolateBench");

                tensor
            }
        }
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<R>, String> {
        use cubek::interpolate::{interpolate, interpolate_backward};

        match &self.problem {
            InterpolateProblem::Forward(prob) => {
                let [n, _, _, c] = prob.input_shape;
                let output_shape = vec![n, prob.output_size[0], prob.output_size[1], c];
                let output = TensorHandle::empty(&self.client, output_shape, self.dtype);

                interpolate(
                    &self.client,
                    input.binding(),
                    output.clone().binding(),
                    prob.options.clone(),
                    self.dtype,
                )
                .map_err(|err| format!("{err}"))?;

                Ok(output)
            }
            InterpolateProblem::Backward(prob) => {
                let [n, h, w, c] = prob.out_grad_shape;
                let input_grad_shape = vec![n, h, w, c];

                // Random input tensor for backward pass. The actual values don't matter for benchmarking, so we can just fill it with random data.
                let backward_input =
                    TensorHandle::empty(&self.client, input_grad_shape.clone(), self.dtype);
                random_uniform(
                    &self.client,
                    -1.,
                    1.,
                    backward_input.clone().binding(),
                    self.dtype,
                )
                .expect("Failed to initialize random values for backward input");

                let output = TensorHandle::empty(&self.client, input_grad_shape, self.dtype);

                interpolate_backward(
                    &self.client,
                    backward_input.binding(),
                    input.clone().binding(), // The input to backward is the output gradient, which has the same shape as the forward output.
                    output.clone().binding(),
                    prob.options.clone(),
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
