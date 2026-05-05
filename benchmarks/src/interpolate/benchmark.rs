use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};

use crate::{
    interpolate::{
        problem::{InterpolateProblem, problem_for},
        strategy::strategy_for,
    },
    registry::RunSamples,
};

use cubek_interpolate::{interpolate, interpolate_options::InterpolateOptions};

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
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = InterpolateBench::<R> {
        problem,
        strategy,
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
    strategy: InterpolateOptions,
    device: R::Device,
    client: ComputeClient<R>,
    dtype: StorageType,
    samples: usize,
}

impl<R: Runtime> Benchmark for InterpolateBench<R> {
    type Input = TensorHandle<R>;
    type Output = TensorHandle<R>;

    fn prepare(&self) -> Self::Input {
        TensorHandle::empty(&self.client, self.problem.input_shape.to_vec(), self.dtype)
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<R>, String> {
        Ok(interpolate(
            &self.client,
            input.binding(),
            self.problem.output_size,
            self.strategy.clone(),
            self.dtype,
        ))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "interpolate-{:?}-{:?}-{:?}-{:?}",
            self.dtype, self.strategy.mode, self.device, self.problem.input_shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
