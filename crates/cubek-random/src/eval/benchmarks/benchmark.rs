use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek_test_utils::RunSamples;

use crate::eval::benchmarks::problem::{Distribution, RandomProblem};
use crate::eval::benchmarks::strategy::RandomStrategy;
use crate::{random_bernoulli, random_normal, random_uniform};

pub fn bench(
    _strategy: &RandomStrategy,
    problem: &RandomProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = RandomBench {
        shape: problem.shape.clone(),
        distribution: problem.distribution,
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

struct RandomBench {
    shape: Vec<usize>,
    distribution: Distribution,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
}

impl Benchmark for RandomBench {
    type Input = TensorHandle<TestRuntime>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        TensorHandle::empty(&self.client, self.shape.clone(), f32::elem_type_native())
    }

    fn execute(&self, out: Self::Input) -> Result<(), String> {
        let dtype = f32::elem_type_native();
        let out = out.binding();

        match self.distribution {
            Distribution::Uniform(lower_bound, upper_bound) => {
                random_uniform(&self.client, lower_bound, upper_bound, out, dtype)
            }
            Distribution::Normal(mean, std) => random_normal(&self.client, mean, std, out, dtype),
            Distribution::Bernoulli(probability) => {
                random_bernoulli(&self.client, probability, out, dtype)
            }
        }
        .map_err(|e| format!("{e}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = <TestRuntime as Runtime>::client(&self.device);

        format!(
            "random-{}-{}-{:?}",
            <TestRuntime as Runtime>::name(&client),
            self.distribution.name(),
            self.shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .clone()
            .profile(|| self.execute(args), "random-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
