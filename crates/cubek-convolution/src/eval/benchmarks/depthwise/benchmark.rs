//! Timing one depthwise problem under one tiling.

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::{ConvolutionArgs, DepthwiseStrategy, DepthwiseTensors, launch_depthwise};

use super::problem::DepthwiseProblem;

pub fn bench(
    strategy: &DepthwiseStrategy,
    problem: &DepthwiseProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let bench = DepthwiseBench {
        problem: *problem,
        strategy: *strategy,
        client,
        device,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct DepthwiseBench {
    problem: DepthwiseProblem,
    strategy: DepthwiseStrategy,
    device: cubecl::Device,
    client: Client,
    samples: usize,
}

/// The element the model runs this convolution in.
fn dtype() -> ElemType {
    f32::elem_type_native()
}

fn uniform(client: &Client, shape: [usize; 4], seed: u64) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype())
        .uniform(seed, 0.0, 1.0)
        .generate_without_host_data()
}

impl Benchmark for DepthwiseBench {
    type Input = (TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        (
            uniform(&self.client, self.problem.in_shape(), 0),
            uniform(&self.client, self.problem.weight_shape(), 1),
        )
    }

    fn execute(&self, (input, weight): Self::Input) -> Result<(), String> {
        let problem = &self.problem;
        let out: TensorHandle =
            TensorHandle::empty(&self.client, problem.out_shape().to_vec(), dtype());

        let padding = problem.padding();
        let args = ConvolutionArgs::<2> {
            stride: [problem.stride; 2],
            padding: [padding; 2],
            dilation: [problem.dilation; 2],
        };

        launch_depthwise(
            &self.client,
            DepthwiseTensors {
                input: input.binding(),
                weight: weight.binding(),
                out: out.binding(),
            },
            args,
            problem.channels,
            dtype(),
            self.strategy,
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = self.device.client();
        format!("{}-depthwise-{}", client.name(), dtype()).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        let (launched, duration) = self
            .client
            .profile(|| self.execute(args), "depthwise-bench")
            .map_err(|it| format!("{it:?}"))?;
        launched.map(|_| duration)
    }
}
