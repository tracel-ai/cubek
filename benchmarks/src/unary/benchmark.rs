use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    calculate_cube_count_elemwise,
    client::ComputeClient,
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::random::random_uniform;

use crate::{
    registry::RunSamples,
    unary::{
        problem::{UnaryProblem, problem_for},
        strategy::{UnaryStrategy, strategy_for},
    },
};

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

pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime, f32>(Default::default(), strategy_id, problem_id, num_samples)
}

pub fn run_on<R: Runtime, E: frontend::Float>(
    device: R::Device,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = UnaryBench::<R, E> {
        problem,
        strategy,
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

struct UnaryBench<R: Runtime, E> {
    problem: UnaryProblem,
    strategy: UnaryStrategy,
    device: R::Device,
    client: ComputeClient<R>,
    samples: usize,
    _e: PhantomData<E>,
}

impl<R: Runtime, E: Float> Benchmark for UnaryBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let lhs = TensorHandle::empty(&client, self.problem.shape.clone(), elem);
        random_uniform(&client, 0., 1., lhs.clone().binding(), elem.storage_type()).unwrap();
        let rhs = TensorHandle::empty(&client, self.problem.shape.clone(), elem);
        random_uniform(&client, 0., 1., rhs.clone().binding(), elem.storage_type()).unwrap();
        let out = TensorHandle::empty(&client, self.problem.shape.clone(), elem);
        random_uniform(&client, 0., 1., out.clone().binding(), elem.storage_type()).unwrap();

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Input) -> Result<(), String> {
        let num_elems = out.shape().num_elements();

        let working_units = num_elems / self.strategy.vectorization;
        let cube_dim = CubeDim::new(&self.client, working_units);
        let cube_count = calculate_cube_count_elemwise(&self.client, working_units, cube_dim);

        execute::launch::<E, R>(
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
        let client = R::client(&self.device);

        format!(
            "unary-{}-{}-{:?}",
            R::name(&client),
            E::as_type_native_unchecked(),
            self.strategy.vectorization,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .clone()
            .profile(|| self.execute(args), "unary-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
