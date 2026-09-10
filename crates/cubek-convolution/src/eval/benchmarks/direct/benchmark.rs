use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::{ConvolutionArgs, DirectTensors, launch_direct};

use super::problem::{DirectProblem, DirectStrategy};

pub fn bench(
    strategy: &DirectStrategy,
    problem: &DirectProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let bench = DirectBench {
        problem: *problem,
        strategy: *strategy,
        client,
        device,
        samples: num_samples,
    };

    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::System))
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct DirectBench {
    problem: DirectProblem,
    #[allow(dead_code)]
    strategy: DirectStrategy,
    device: cubecl::Device,
    client: Client,
    samples: usize,
}

fn dtype() -> ElemType {
    f32::elem_type_native()
}

fn uniform(client: &Client, shape: [usize; 4], seed: u64) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype())
        .uniform(seed, 0.0, 1.0)
        .generate_without_host_data()
}

impl Benchmark for DirectBench {
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
            dilation: [1; 2],
        };

        launch_direct::<2>(
            &self.client,
            DirectTensors {
                input: input.binding(),
                weight: weight.binding(),
                bias: None,
                out: out.binding(),
            },
            args,
            1,
            dtype(),
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = self.device.client();
        format!("{}-direct-conv-{}", client.name(), dtype()).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "direct-conv-bench", || self.execute(args))
    }
}
