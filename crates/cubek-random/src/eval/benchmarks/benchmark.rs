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
use crate::{
    PrngStrategy, random_bernoulli_with_strategy, random_normal_with_strategy,
    random_uniform_with_strategy,
};

pub fn bench(
    strategy: &RandomStrategy,
    problem: &RandomProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = RandomBench {
        shape: problem.shape.clone(),
        distribution: problem.distribution,
        strategy: strategy.prng,
        client: client.clone(),
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
    strategy: PrngStrategy,
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
        let client = &self.client;
        let strategy = self.strategy;

        match self.distribution {
            Distribution::Uniform(lower_bound, upper_bound) => {
                random_uniform_with_strategy(client, lower_bound, upper_bound, out, dtype, strategy)
            }
            Distribution::Normal(mean, std) => {
                random_normal_with_strategy(client, mean, std, out, dtype, strategy)
            }
            Distribution::Bernoulli(probability) => {
                random_bernoulli_with_strategy(client, probability, out, dtype, strategy)
            }
        }
        .map_err(|e| format!("{e}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "random-{}-{}-{:?}",
            <TestRuntime as Runtime>::name(&self.client),
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
