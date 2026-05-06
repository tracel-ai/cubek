use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};

use crate::{
    contiguous::{problem::ContiguousProblem, strategy::ContiguousStrategy},
    registry::RunSamples,
};

pub fn bench(
    _strategy: &ContiguousStrategy,
    problem: &ContiguousProblem,
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
    problem: &ContiguousProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);

    let bench = IntoContiguousBench::<R> {
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

struct IntoContiguousBench<R: Runtime> {
    shape: Vec<usize>,
    dims: Vec<(usize, usize)>,
    device: R::Device,
    client: ComputeClient<R>,
    dtype: StorageType,
    samples: usize,
}

impl<R: Runtime> Benchmark for IntoContiguousBench<R> {
    type Input = TensorHandle<R>;
    type Output = TensorHandle<R>;

    fn prepare(&self) -> Self::Input {
        let mut handle = TensorHandle::empty(&self.client, self.shape.clone(), self.dtype);
        for (dim0, dim1) in self.dims.iter() {
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
            self.dtype, self.dims, self.device, self.shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
