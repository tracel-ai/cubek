use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};
use cubek_tile::Residence;

use crate::{
    definition::InterpolateProblem, interpolate, interpolate_backward, launch::InterpolateStrategy,
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
        return Err("interpolation shared memory strategy is not supported on CPU".to_string());
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

        if !is_cpu && config.cols_per_lane > 1 {
            return Err("tile interpolation with c > 1 is inefficient on GPU devices".to_string());
        }

        if let InterpolateProblem::Forward(prob) = problem
            && config.input_residence == Residence::Smem
        {
            let geometry = config.geometry(prob.channels, lanes);
            let (row, col) = (
                crate::launch::tile::coordinate::Rational::of(crate::definition::get_transform(
                    prob.input_height,
                    prob.output_height,
                    prob.options,
                )),
                crate::launch::tile::coordinate::Rational::of(crate::definition::get_transform(
                    prob.input_width,
                    prob.output_width,
                    prob.options,
                )),
            );
            let taps = match prob.options.mode {
                crate::definition::InterpolateMode::Nearest(_) => 1,
                crate::definition::InterpolateMode::Bilinear => 2,
                crate::definition::InterpolateMode::Bicubic => 4,
                crate::definition::InterpolateMode::Lanczos3 => 6,
            };
            let radius = (taps - 1) / 2;
            let vector_size = 1;
            let requested_smem = crate::launch::tile::space::stage_window_bytes(
                row,
                col,
                taps,
                radius,
                geometry,
                vector_size,
                dtype.size(),
            );
            if requested_smem > hardware.max_shared_memory_size {
                return Err(format!(
                    "requested shared memory {requested_smem} bytes exceeds device limit of {} bytes",
                    hardware.max_shared_memory_size
                ));
            }
        }
    }

    let bench = InterpolateBench {
        problem: problem.clone(),
        strategy: *strategy,
        device,
        client,
        dtype,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
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
}
