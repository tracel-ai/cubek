use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{random::random_uniform, reduce::launch::ReduceStrategy};

use crate::{reduce::problem::ReduceProblem, registry::RunSamples};

pub fn bench(
    strategy: &ReduceStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    bench_on::<cubecl::TestRuntime, f32>(Default::default(), strategy, problem, num_samples)
}

pub fn bench_on<R: Runtime, E: frontend::Float>(
    device: R::Device,
    strategy: &ReduceStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);

    let bench = ReduceBench::<R, E> {
        shape: problem.shape.clone(),
        axis: problem.axis,
        config: problem.config,
        strategy: strategy.clone(),
        device,
        client,
        samples: num_samples,
        _e: PhantomData,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct ReduceBench<R: Runtime, E> {
    shape: Vec<usize>,
    axis: usize,
    config: cubek::reduce::components::instructions::ReduceOperationConfig,
    strategy: ReduceStrategy,
    device: R::Device,
    client: ComputeClient<R>,
    samples: usize,
    _e: PhantomData<E>,
}

impl<R: Runtime, E: Float> Benchmark for ReduceBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let input = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(
            &client,
            0.,
            1.,
            input.clone().binding(),
            elem.storage_type(),
        )
        .unwrap();
        let mut shape_out = self.shape.clone();
        let reduce_len = match self.config {
            cubek::reduce::components::instructions::ReduceOperationConfig::ArgTopK(len) => len,
            cubek::reduce::components::instructions::ReduceOperationConfig::TopK(len) => len,
            _ => 1,
        };
        shape_out[self.axis] = reduce_len;
        let out = TensorHandle::empty(&client, shape_out, elem);

        (input, out)
    }

    fn execute(&self, (input, out): Self::Input) -> Result<(), String> {
        cubek::reduce::reduce::<R>(
            &self.client,
            input.binding(),
            out.binding(),
            self.axis,
            self.strategy.clone(),
            self.config,
            cubek::reduce::ReduceDtypes {
                input: E::as_type_native_unchecked().storage_type(),
                output: E::as_type_native_unchecked().storage_type(),
                accumulation: f32::as_type_native_unchecked().storage_type(),
            },
        )
        .map_err(|err| format!("{err}"))?;

        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}-{:?}-{:?}",
            self.axis,
            E::as_type_native_unchecked(),
            self.shape,
            self.strategy,
            self.config,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
