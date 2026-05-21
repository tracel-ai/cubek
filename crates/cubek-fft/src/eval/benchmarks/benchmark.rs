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

use crate::eval::benchmarks::problem::FftProblem;
use crate::eval::benchmarks::strategy::FftStrategy;
use crate::{FftMode, irfft_launch, rfft_launch};

pub fn bench(
    _strategy: &FftStrategy,
    problem: &FftProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = FftBench::<f32> {
        shape: problem.shape.clone(),
        mode: problem.mode,
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

struct FftBench<E> {
    shape: Vec<usize>,
    mode: FftMode,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

#[derive(Clone)]
struct FftInput {
    signal: TensorHandle<TestRuntime>,
    spectrum_re: TensorHandle<TestRuntime>,
    spectrum_im: TensorHandle<TestRuntime>,
}

fn make_uniform(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::from(shape))
        .layout(StridedLayout::RowMajor)
        .dtype(dtype)
        .uniform(seed, 0., 1.)
        .generate_without_host_data()
}

fn empty_handle(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    elem: cubecl::ir::Type,
) -> TensorHandle<TestRuntime> {
    TensorHandle::empty(client, shape, elem)
}

impl<E: Float> Benchmark for FftBench<E> {
    type Input = FftInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let elem = E::as_type_native_unchecked();
        let storage = elem.storage_type();

        let mut shape_out = self.shape.clone();
        let dim = self.shape.len() - 1;
        shape_out[dim] = self.shape[dim] / 2 + 1;

        match self.mode {
            FftMode::Forward => {
                let signal = make_uniform(&client, self.shape.clone(), storage, 0);
                let spectrum_re = empty_handle(&client, shape_out.clone(), elem);
                let spectrum_im = empty_handle(&client, shape_out, elem);
                FftInput {
                    signal,
                    spectrum_re,
                    spectrum_im,
                }
            }
            FftMode::Inverse => {
                let signal = empty_handle(&client, self.shape.clone(), elem);
                let spectrum_re = make_uniform(&client, shape_out.clone(), storage, 0);
                let spectrum_im = make_uniform(&client, shape_out, storage, 1);
                FftInput {
                    signal,
                    spectrum_re,
                    spectrum_im,
                }
            }
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim = self.shape.len() - 1;
        match self.mode {
            FftMode::Forward => rfft_launch(
                &self.client,
                input.signal.binding(),
                input.spectrum_re.binding(),
                input.spectrum_im.binding(),
                dim,
                E::as_type_native_unchecked().storage_type(),
            )
            .map_err(|err| format!("{err}"))?,
            FftMode::Inverse => irfft_launch(
                &self.client,
                input.spectrum_re.binding(),
                input.spectrum_im.binding(),
                input.signal.binding(),
                dim,
                E::as_type_native_unchecked().storage_type(),
            )
            .map_err(|err| format!("{err}"))?,
        }
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "fft-{}-{:?}-{:?}",
            E::as_type_native_unchecked(),
            self.shape,
            self.mode,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
