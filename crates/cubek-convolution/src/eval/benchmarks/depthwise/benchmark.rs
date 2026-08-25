//! Timing one depthwise problem under one tiling.

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::{ConvolutionArgs, DepthwiseTiling, launch_depthwise_tiled};

use super::DepthwiseStrategy;

use super::problem::DepthwiseProblem;

pub fn bench(
    strategy: &DepthwiseStrategy,
    problem: &DepthwiseProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let tiling = match *strategy {
        DepthwiseStrategy::Fixed(tiling) => tiling,
        DepthwiseStrategy::Routine => DepthwiseTiling::for_problem(
            problem.channels,
            problem.kernel * problem.kernel,
            client.properties().hardware.plane_size_max as usize,
        ),
    };

    let bench = DepthwiseBench {
        problem: *problem,
        tiling,
        client,
        device,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations).with_flops(problem.flops()))
}

struct DepthwiseBench {
    problem: DepthwiseProblem,
    tiling: DepthwiseTiling,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
}

/// The element the model runs this convolution in.
fn dtype() -> ElemType {
    f32::elem_type_native()
}

fn uniform(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 4],
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype())
        .uniform(seed, 0.0, 1.0)
        .generate_without_host_data()
}

impl Benchmark for DepthwiseBench {
    type Input = (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        (
            uniform(&self.client, self.problem.in_shape(), 0),
            uniform(&self.client, self.problem.weight_shape(), 1),
        )
    }

    fn execute(&self, (input, weight): Self::Input) -> Result<(), String> {
        let problem = &self.problem;
        let out: TensorHandle<TestRuntime> =
            TensorHandle::empty(&self.client, problem.out_shape().to_vec(), dtype());

        let padding = problem.padding();
        let args = ConvolutionArgs::<2> {
            stride: [problem.stride; 2],
            padding: [padding; 2],
            dilation: [problem.dilation; 2],
        };

        launch_depthwise_tiled::<TestRuntime>(
            &self.client,
            input.binding(),
            weight.binding(),
            out.binding(),
            &problem.in_shape(),
            &problem.weight_shape(),
            &problem.out_shape(),
            args,
            problem.channels,
            dtype(),
            self.tiling,
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = <TestRuntime as Runtime>::client(&self.device);
        format!(
            "{}-depthwise-{}",
            <TestRuntime as Runtime>::name(&client),
            dtype()
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "depthwise-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
