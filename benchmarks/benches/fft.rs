use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    fft::{FftMode, irfft_launch, rfft_launch},
    random::random_uniform,
};
use std::marker::PhantomData;

#[allow(dead_code)]
struct FftBench<R: Runtime, E> {
    shape: Vec<usize>,
    device: R::Device,
    client: ComputeClient<R>,
    fft_mode: FftMode,
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
        // Todo: generalize dim
        shape_out[2] = self.shape[2] / 2 + 1;

        let spectrum_re = TensorHandle::empty(&client, shape_out.clone(), elem);
        let spectrum_im = TensorHandle::empty(&client, shape_out, elem);

        match self.fft_mode {
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

    fn execute(&self, fft_input: Self::Input) -> Result<(), String> {
        let signal = fft_input.signal;
        let spectrum_re = fft_input.spectrum_re;
        let spectrum_im = fft_input.spectrum_im;
        let dim = self.shape.len() - 1;
        match self.fft_mode {
            FftMode::Forward => rfft_launch(
                &self.client,
                signal.binding(),
                spectrum_re.binding(),
                spectrum_im.binding(),
                dim,
                E::as_type_native_unchecked().storage_type(),
            )
            .map_err(|err| format!("{err}"))?,
            FftMode::Inverse => irfft_launch(
                &self.client,
                spectrum_re.binding(),
                spectrum_im.binding(),
                signal.binding(),
                dim,
                E::as_type_native_unchecked().storage_type(),
            )
            .map_err(|err| format!("{err}"))?,
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "fft{:?}-{:?}-{:?}",
            E::as_type_native_unchecked(),
            self.shape,
            self.fft_mode,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: R::Device) {
    let client = R::client(&device);

    let modes = [FftMode::Forward, FftMode::Inverse];
    for fft_mode in modes {
        let bench = FftBench {
            shape: [5, 2, 2048].to_vec(),
            device: device.clone(),
            fft_mode,
            client: client.clone(),
            _e: PhantomData::<E>,
        };
        match bench.run(TimingMethod::System) {
            Ok(val) => {
                print!("Name: {}", bench.name());
                println!("{val}\n");
            }
            Err(err) => println!("Can't run the benchmark: {err}"),
        }
    }
}

fn main() {
    run::<cubecl::TestRuntime, f32>(Default::default());
}
