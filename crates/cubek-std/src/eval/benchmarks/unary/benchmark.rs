use std::marker::PhantomData;

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    calculate_cube_count_elemwise,
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::eval::benchmarks::unary::problem::UnaryProblem;
use crate::eval::benchmarks::unary::strategy::UnaryStrategy;

#[cube(launch)]
fn execute<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>) {
    if ABSOLUTE_POS < out.len() {
        for i in 0..256u32 {
            if i % 2 == 0 {
                out[ABSOLUTE_POS] -= F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            } else {
                out[ABSOLUTE_POS] += F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            }
        }
    }
}

pub fn bench(
    strategy: &UnaryStrategy,
    problem: &UnaryProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let bench = UnaryBench::<f32> {
        shape: problem.shape.clone(),
        vectorization: strategy.vectorization,
        client,
        device,
        samples: num_samples,
        _e: PhantomData,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct UnaryBench<E> {
    shape: Vec<usize>,
    vectorization: VectorSize,
    device: cubecl::Device,
    client: Client,
    samples: usize,
    _e: PhantomData<E>,
}

impl<E: Float> Benchmark for UnaryBench<E> {
    type Input = (TensorHandle, TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = self.device.client();
        let storage = E::elem_type_native();

        let make = |seed: u64| -> TensorHandle {
            TestInput::builder(client.clone(), Shape::from(self.shape.clone()))
                .dtype(storage)
                .uniform(seed, 0., 1.)
                .generate_without_host_data()
        };

        let lhs = make(0);
        let rhs = make(1);
        let out = make(2);

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Input) -> Result<(), String> {
        let num_elems = out.shape().num_elements();

        let working_units = num_elems / self.vectorization;
        let cube_dim = CubeDim::new(&self.client, working_units);
        let cube_count = calculate_cube_count_elemwise(&self.client, working_units, cube_dim);

        execute::launch::<E>(
            &self.client,
            cube_count,
            cube_dim,
            lhs.into_arg(),
            rhs.into_arg(),
            out.into_arg(),
        );

        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = self.device.client();

        format!(
            "unary-{}-{}-{:?}",
            client.name(),
            E::elem_type_native(),
            self.vectorization,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        let (launched, duration) = self
            .client
            .clone()
            .profile(|| self.execute(args), "unary-bench")
            .map_err(|it| format!("{it:?}"))?;
        launched.map(|_| duration)
    }
}
