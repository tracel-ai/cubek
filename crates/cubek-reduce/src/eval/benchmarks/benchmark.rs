use std::marker::PhantomData;

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

use crate::ReduceStrategy;
use crate::components::instructions::ReduceOperationConfig;
use crate::eval::benchmarks::problem::ReduceProblem;

pub fn bench(
    strategy: &ReduceStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = ReduceBench::<f32> {
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

struct ReduceBench<E> {
    shape: Vec<usize>,
    axis: usize,
    config: ReduceOperationConfig,
    strategy: ReduceStrategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

impl<E: Float> Benchmark for ReduceBench<E> {
    type Input = (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let elem = E::as_type_native_unchecked();
        let storage = elem.storage_type();

        let input = TestInput::builder(client.clone(), Shape::from(self.shape.clone()))
            .dtype(storage)
            .uniform(0, 0., 1.)
            .generate_without_host_data();
        let mut shape_out = self.shape.clone();
        let reduce_len = match self.config {
            ReduceOperationConfig::ArgTopK(len) => len,
            ReduceOperationConfig::TopK(len) => len,
            _ => 1,
        };
        shape_out[self.axis] = reduce_len;
        let out = TensorHandle::empty(&client, shape_out, elem);

        (input, out)
    }

    fn execute(&self, (input, out): Self::Input) -> Result<(), String> {
        crate::reduce::<TestRuntime>(
            &self.client,
            input.binding(),
            out.binding(),
            self.axis,
            self.strategy.clone(),
            self.config,
            crate::ReduceDtypes {
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
