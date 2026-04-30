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
    fft::{
        problem::{FftProblem, problem_for},
        strategy::strategy_for,
    },
    registry::RunSamples,
};

pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime, f32>(Default::default(), strategy_id, problem_id, num_samples)
}

pub fn run_on<R: Runtime, E: frontend::Float>(
    device: R::Device,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = FftBench::<R, E> {
        problem,
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
    problem: FftProblem,
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

        let signal = TensorHandle::empty(&client, self.problem.shape.clone(), elem);

        let mut shape_out = self.problem.shape.clone();
        shape_out[2] = self.problem.shape[2] / 2 + 1;

        let spectrum_re = TensorHandle::empty(&client, shape_out.clone(), elem);
        let spectrum_im = TensorHandle::empty(&client, shape_out, elem);

        match self.problem.mode {
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
        let dim = self.problem.shape.len() - 1;
        match self.problem.mode {
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
            self.problem.shape,
            self.problem.mode,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
