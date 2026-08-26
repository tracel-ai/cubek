use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    std::throughput::measure_peak_throughput,
    throughput::{ThroughputKey, ThroughputMode},
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};
use cubek_tile::Residence;

use crate::{
    definition::{InterpolateCost, InterpolateProblem},
    interpolate, interpolate_backward,
    launch::InterpolateStrategy,
    launch::interpolate_tile_launch,
};

use super::InterpolateBenchmarkStrategy;

pub fn bench(
    strategy: &InterpolateBenchmarkStrategy,
    problem: &InterpolateProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let hardware = &client.properties().hardware;
    let is_cpu = hardware.num_cpu_cores.is_some();

    if is_cpu
        && (matches!(
            strategy,
            InterpolateBenchmarkStrategy::Standard(InterpolateStrategy::SharedMemoryStrategy(_))
        ) || matches!(
            strategy,
            InterpolateBenchmarkStrategy::Tile(config) if config.input_residence == Residence::Smem
        ))
    {
        return Err("interpolation shared memory strategy is not used on CPU".to_string());
    }

    let dtype = f32::elem_type_native();

    if let InterpolateBenchmarkStrategy::Tile(config) = strategy {
        let lanes = hardware.plane_size_max as usize;
        let planes = config.planes_per_cube;
        let units_per_cube = planes * lanes;
        if units_per_cube > hardware.max_units_per_cube as usize {
            return Err(format!(
                "tile units per cube ({units_per_cube}) exceeds device max ({})",
                hardware.max_units_per_cube
            ));
        }
    }

    let bench = InterpolateBench {
        problem: problem.clone(),
        strategy: *strategy,
        device,
        client: client.clone(),
        dtype,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    let work = InterpolateCost::new(problem.clone(), dtype).work();

    Ok(RunSamples::new(durations)
        .with_flops(work.compute_ops as f64, None)
        .with_bytes(work.bytes, memory_peak_bytes_per_s(&client)))
}

/// The device's measured copy peak, in bytes/s, for the resampling to be judged against.
///
/// [`ThroughputMode::Memory`] is a copy, which is the access interpolation performs: it
/// reads one tensor and writes another. `measure_peak_throughput` caches per device, so
/// this does not need to as well.
fn memory_peak_bytes_per_s(client: &ComputeClient<TestRuntime>) -> Option<f64> {
    let key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };
    let peak = measure_peak_throughput(client, key).bytes_per_s(&key);

    (peak > 0.0).then_some(peak)
}

struct InterpolateBench {
    problem: InterpolateProblem,
    strategy: InterpolateBenchmarkStrategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    dtype: ElemType,
    samples: usize,
}

impl Benchmark for InterpolateBench {
    type Input = TensorHandle<TestRuntime>;
    type Output = TensorHandle<TestRuntime>;

    fn prepare(&self) -> Self::Input {
        let shape = match &self.problem {
            InterpolateProblem::Forward(prob) => prob.input_shape(),
            InterpolateProblem::Backward(prob) => Shape::new(prob.out_grad_shape),
        };
        TestInput::builder(self.client.clone(), shape)
            .dtype(self.dtype)
            .uniform(0, -1., 1.)
            .generate_without_host_data()
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<TestRuntime>, String> {
        match &self.problem {
            InterpolateProblem::Forward(prob) => {
                let output = TensorHandle::empty(&self.client, prob.output_shape(), self.dtype);

                match self.strategy {
                    InterpolateBenchmarkStrategy::Standard(strategy) => interpolate(
                        &self.client,
                        input.binding(),
                        output.clone().binding(),
                        prob.options,
                        strategy,
                        self.dtype,
                    ),
                    InterpolateBenchmarkStrategy::Tile(config) => interpolate_tile_launch(
                        &self.client,
                        input.binding(),
                        output.clone().binding(),
                        prob.options,
                        self.dtype,
                        config,
                    ),
                }
                .map_err(|err| format!("{err}"))?;

                Ok(output)
            }
            InterpolateProblem::Backward(prob) => {
                if matches!(self.strategy, InterpolateBenchmarkStrategy::Tile(_)) {
                    return Err("tile interpolation does not support backward problems".to_string());
                }
                let [n, _, _, c] = prob.out_grad_shape;
                let input_grad_shape = vec![n, prob.input_size[0], prob.input_size[1], c];

                // Random input tensor for the backward pass. The actual values don't matter
                // for benchmarking, so we just fill it with random data.
                let backward_input =
                    TestInput::builder(self.client.clone(), input_grad_shape.clone())
                        .dtype(self.dtype)
                        .uniform(0, -1., 1.)
                        .generate_without_host_data();

                let output = TensorHandle::empty(&self.client, input_grad_shape, self.dtype);

                interpolate_backward(
                    &self.client,
                    backward_input.binding(),
                    input.clone().binding(),
                    output.clone().binding(),
                    prob.options,
                    self.dtype,
                )
                .map_err(|err| format!("{err}"))?;

                Ok(output)
            }
        }
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        match &self.problem {
            InterpolateProblem::Forward(prob) => format!(
                "interpolate-{:?}-{:?}-{:?}-{:?}-{:?}",
                self.strategy,
                self.dtype,
                prob.options.mode,
                self.device,
                prob.input_shape(),
            )
            .to_lowercase(),
            InterpolateProblem::Backward(prob) => format!(
                "interpolate-backward-{:?}-{:?}-{:?}-{:?}-{:?}",
                self.strategy, self.dtype, prob.options.mode, self.device, prob.out_grad_shape,
            )
            .to_lowercase(),
        }
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    /// Measure with device timestamps around the launch, so the reported duration is the
    /// kernel's rather than the host's view of launch, output allocation and sync.
    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "interpolate-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}
