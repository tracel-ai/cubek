use std::sync::OnceLock;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    std::throughput::measure_peak_throughput,
    throughput::{ThroughputKey, ThroughputMode},
};
use cubek_test_utils::RunSamples;

use crate::eval::benchmarks::problem::{Distribution, RandomProblem};
use crate::eval::benchmarks::strategy::RandomStrategy;
use crate::{
    Bernoulli, BernoulliFamily, Normal, NormalFamily, PrngStrategy, Uniform, UniformFamily, random,
};

pub fn bench(
    strategy: &RandomStrategy,
    problem: &RandomProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let dtype = f32::elem_type_native();
    let bytes: usize = problem.shape.iter().product::<usize>() * dtype.size();

    let bench = RandomBench {
        shape: problem.shape.clone(),
        distribution: problem.distribution,
        strategy: strategy.prng,
        client: client.clone(),
        device,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations).with_bytes(bytes, write_peak_bytes_per_s(&client)))
}

/// The device's measured write peak, in bytes/s, for a random kernel to be
/// judged against.
///
/// The throughput facility has no write-only probe, so this uses
/// [`ThroughputMode::Memory`], a copy: the mode that issues stores, though half
/// its traffic is a read `random` never does. Cached process-wide because
/// re-measuring re-allocates the probe's buffers.
fn write_peak_bytes_per_s(client: &ComputeClient<TestRuntime>) -> Option<f64> {
    static PEAK: OnceLock<f64> = OnceLock::new();

    let key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };
    let peak = *PEAK.get_or_init(|| measure_peak_throughput(client, key).bytes_per_s(&key));

    (peak > 0.0).then_some(peak)
}

struct RandomBench {
    shape: Vec<usize>,
    distribution: Distribution,
    strategy: PrngStrategy,
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
        let client = &self.client;
        let strategy = self.strategy;

        match self.distribution {
            Distribution::Uniform(lower_bound, upper_bound) => random::<UniformFamily, _>(
                client,
                Uniform {
                    lower_bound,
                    upper_bound,
                },
                out,
                dtype,
                strategy,
            ),
            Distribution::Normal(mean, std) => {
                random::<NormalFamily, _>(client, Normal { mean, std }, out, dtype, strategy)
            }
            Distribution::Bernoulli(probability) => random::<BernoulliFamily, _>(
                client,
                Bernoulli { probability },
                out,
                dtype,
                strategy,
            ),
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
