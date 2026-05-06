use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    fft::{FftMode, irfft_launch, rfft_launch},
    random::random_uniform,
};

use crate::{
    fft::{problem::FftProblem, strategy::FftStrategy},
    registry::RunSamples,
};

pub fn bench(
    strategy: &FftStrategy,
    problem: &FftProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    bench_on::<cubecl::TestRuntime, f32>(Default::default(), strategy, problem, num_samples)
}

pub fn bench_on<R: Runtime, E: frontend::Float>(
    device: R::Device,
    _strategy: &FftStrategy,
    problem: &FftProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);

    let bench = FftBench::<R, E> {
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

struct FftBench<R: Runtime, E> {
    shape: Vec<usize>,
    mode: FftMode,
    device: R::Device,
    client: ComputeClient<R>,
    samples: usize,
    _e: PhantomData<E>,
}

#[derive(Clone)]
struct FftInput<R: Runtime> {
    signal: TensorHandle<R>,
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
}

impl<R: Runtime, E: Float> Benchmark for FftBench<R, E> {
    type Input = FftInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let signal = TensorHandle::empty(&client, self.shape.clone(), elem);

        let mut shape_out = self.shape.clone();
        let dim = self.shape.len() - 1;
        shape_out[dim] = self.shape[dim] / 2 + 1;

        let spectrum_re = TensorHandle::empty(&client, shape_out.clone(), elem);
        let spectrum_im = TensorHandle::empty(&client, shape_out, elem);

        match self.mode {
            FftMode::Forward => {
                random_uniform(
                    &client,
                    0.,
                    1.,
                    signal.clone().binding(),
                    elem.storage_type(),
                )
                .unwrap();
            }
            FftMode::Inverse => {
                random_uniform(
                    &client,
                    0.,
                    1.,
                    spectrum_re.clone().binding(),
                    elem.storage_type(),
                )
                .unwrap();
                random_uniform(
                    &client,
                    0.,
                    1.,
                    spectrum_im.clone().binding(),
                    elem.storage_type(),
                )
                .unwrap();
            }
        };
        FftInput {
            signal,
            spectrum_re,
            spectrum_im,
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
