use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};

use crate::{
    contiguous::{
        problem::{ContiguousProblem, problem_for},
        strategy::strategy_for,
    },
    registry::RunSamples,
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
    strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = IntoContiguousBench::<R> {
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

struct IntoContiguousBench<R: Runtime> {
    problem: ContiguousProblem,
    device: R::Device,
    client: ComputeClient<R>,
    dtype: StorageType,
    samples: usize,
}

impl<R: Runtime> Benchmark for IntoContiguousBench<R> {
    type Input = TensorHandle<R>;
    type Output = TensorHandle<R>;

    fn prepare(&self) -> Self::Input {
        let mut handle = TensorHandle::empty(&self.client, self.problem.shape.clone(), self.dtype);
        for (dim0, dim1) in self.problem.dims.iter() {
            handle.metadata.swap(*dim0, *dim1);
        }
        handle
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<R>, String> {
        Ok(cubecl::std::tensor::into_contiguous(
            &self.client,
            input.binding(),
            self.dtype,
        ))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "into_contiguous-{:?}-{:?}-{:?}-{:?}",
            self.dtype, self.problem.dims, self.device, self.problem.shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
