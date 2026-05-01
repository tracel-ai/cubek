use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    std::tensor::TensorHandle,
};
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulPrecision},
        launch::{Strategy, launch_ref},
    },
    random::random_uniform,
    std::{InputBinding, MatrixLayout},
};

use crate::{
    gemm::{
        problem::{GemmProblem, Precision, problem_for},
        strategy::strategy_for,
    },
    registry::RunSamples,
};

pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime>(Default::default(), strategy_id, problem_id, num_samples)
}

pub fn run_on<R: Runtime>(
    device: R::Device,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;
    match problem.precision {
        Precision::F32 => run_with::<R, f32>(device, problem, strategy, num_samples),
        Precision::F16 => run_with::<R, half::f16>(device, problem, strategy, num_samples),
    }
}

fn run_with<R: Runtime, MP: MatmulPrecision>(
    device: R::Device,
    problem: GemmProblem,
    strategy: Strategy,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let flops = 2.0 * problem.b as f64 * problem.m as f64 * problem.n as f64 * problem.k as f64;

    let bench = GemmBench::<R> {
        problem,
        strategy,
        client,
        device,
        dtypes: MatmulElems::new_deprecated::<MP>(),
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations).with_flops(flops))
}

struct GemmBench<R: Runtime> {
    problem: GemmProblem,
    strategy: Strategy,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
    samples: usize,
}

impl<R: Runtime> Benchmark for GemmBench<R> {
    type Input = (TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let tl = matches!(self.problem.lhs_layout, MatrixLayout::ColMajor);
        let tr = matches!(self.problem.rhs_layout, MatrixLayout::ColMajor);

        let mut lhs = TensorHandle::empty(
            &client,
            vec![self.problem.b, self.problem.m, self.problem.k],
            self.dtypes.lhs_global,
        );
        if tl {
            let len = lhs.metadata.rank();
            lhs.metadata.strides_mut().swap(len - 2, len - 1);
        }
        random_uniform(
            &client,
            0.0,
            1.0,
            lhs.clone().binding(),
            self.dtypes.lhs_global,
        )
        .unwrap();

        let mut rhs = TensorHandle::empty(
            &client,
            vec![self.problem.b, self.problem.k, self.problem.n],
            self.dtypes.rhs_global,
        );
        if tr {
            let len = rhs.metadata.rank();
            rhs.metadata.strides_mut().swap(len - 2, len - 1);
        }
        random_uniform(
            &client,
            0.0,
            1.1,
            rhs.clone().binding(),
            self.dtypes.rhs_global,
        )
        .unwrap();

        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let client = R::client(&self.device);
        let out = TensorHandle::empty(
            &client,
            vec![self.problem.b, self.problem.m, self.problem.n],
            self.dtypes.acc_global,
        );

        launch_ref(
            &self.strategy,
            &self.client,
            InputBinding::Normal(lhs.binding(), self.dtypes.lhs_global),
            InputBinding::Normal(rhs.binding(), self.dtypes.lhs_global),
            out.clone().binding(),
            &mut self.dtypes.clone(),
        )
        .map_err(|err| format!("{err:?}"))?;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "{}-matmul-Lhs<{}-{}-{}>-Rhs<{}-{}-{}>-{}-{}-{}",
            R::name(&client),
            self.dtypes.lhs_global,
            self.dtypes.lhs_stage,
            self.dtypes.lhs_register,
            self.dtypes.rhs_global,
            self.dtypes.rhs_stage,
            self.dtypes.rhs_register,
            self.dtypes.acc_register,
            self.dtypes.acc_global,
            self.strategy,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "matmul-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}
