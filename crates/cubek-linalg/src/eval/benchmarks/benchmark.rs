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
use cubek_test_utils::{RunSamples, StridedLayout, TestInput};

use crate::eval::benchmarks::problem::QrProblem;
use crate::eval::benchmarks::strategy::QrStrategy;

pub fn bench(
    _strategy: &QrStrategy,
    problem: &QrProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = QrBench::<f32> {
        m: problem.m,
        n: problem.n,
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

struct QrBench<E> {
    m: usize,
    n: usize,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

impl<E: Float + CubeElement> Benchmark for QrBench<E> {
    type Input = TensorHandle<TestRuntime>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let storage = E::as_type_native_unchecked().storage_type();

        // Col-major input: the layout the QR kernels work in, so the timed
        // run measures the factorization rather than a layout conversion.
        TestInput::builder(self.client.clone(), Shape::from(vec![self.m, self.n]))
            .layout(StridedLayout::ColMajor)
            .dtype(storage)
            .uniform(0, -1., 1.)
            .generate_without_host_data()
    }

    fn execute(&self, a: Self::Input) -> Result<(), String> {
        crate::launch::qr::<TestRuntime, E>(&self.client, &a)
            .map(|_| ())
            .map_err(|err| format!("{err:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "qr-{}-baht_tsqr-{}x{}",
            E::as_type_native_unchecked(),
            self.m,
            self.n,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
