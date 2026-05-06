use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};

use crate::interpolate::problem::problem_for;
use crate::interpolate::strategy::strategy_for;
use crate::registry::RunSamples;

use cubek::{
    interpolate::{definition::InterpolateProblem, interpolate},
    random::random_uniform,
};

pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime>(
        Default::default(),
        f32::as_type_native_unchecked().storage_type(),
        strategy_id,
        problem_id,
        num_samples,
    )
}

pub fn run_on<R: Runtime>(
    device: R::Device,
    dtype: StorageType,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let _ = strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = InterpolateBench::<R> {
        problem,
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
        let tensor =
            TensorHandle::empty(&self.client, self.problem.input_shape.to_vec(), self.dtype);

        random_uniform(&self.client, -1., 1., tensor.clone().binding(), self.dtype)
            .expect("Failed to initialize random values for InterpolateBench");

        tensor
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<R>, String> {
        let [n, _, _, c] = self.problem.input_shape;
        let output_shape = vec![
            n,
            self.problem.output_size[0],
            self.problem.output_size[1],
            c,
        ];
        let output = TensorHandle::empty(&self.client, output_shape, self.dtype);

        interpolate(
            &self.client,
            input.binding(),
            output.clone().binding(),
            self.problem.options.clone(),
            self.dtype,
        )
        .map_err(|err| format!("{err}"))?;

        Ok(output)
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "interpolate-{:?}-{:?}-{:?}-{:?}",
            self.dtype, self.problem.options.mode, self.device, self.problem.input_shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
