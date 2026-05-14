use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek_test_utils::RunSamples;

use crate::eval::benchmarks::contiguous::problem::ContiguousProblem;
use crate::eval::benchmarks::contiguous::strategy::ContiguousStrategy;

pub fn bench(
    _strategy: &ContiguousStrategy,
    problem: &ContiguousProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let dtype = f32::as_type_native_unchecked().storage_type();

    let bench = IntoContiguousBench {
        shape: problem.shape.clone(),
        dims: problem.dims.clone(),
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

struct IntoContiguousBench {
    shape: Vec<usize>,
    dims: Vec<(usize, usize)>,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    dtype: StorageType,
    samples: usize,
}

impl Benchmark for IntoContiguousBench {
    type Input = TensorHandle<TestRuntime>;
    type Output = TensorHandle<TestRuntime>;

    fn prepare(&self) -> Self::Input {
        let mut handle = TensorHandle::empty(&self.client, self.shape.clone(), self.dtype);
        for (dim0, dim1) in self.dims.iter() {
            handle.metadata.swap(*dim0, *dim1);
        }
        handle
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<TestRuntime>, String> {
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
            self.dtype, self.dims, self.device, self.shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
