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
    gemm::problem::{GemmProblem, Precision},
    registry::RunSamples,
};

pub fn bench(
    strategy: &Strategy,
    problem: &GemmProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    bench_on::<cubecl::TestRuntime>(Default::default(), strategy, problem, num_samples)
}

pub fn bench_on<R: Runtime>(
    device: R::Device,
    strategy: &Strategy,
    problem: &GemmProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    match problem.precision {
        Precision::F32 => bench_with::<R, f32>(device, problem, strategy, num_samples),
        Precision::F16 => bench_with::<R, half::f16>(device, problem, strategy, num_samples),
    }
}

fn bench_with<R: Runtime, MP: MatmulPrecision>(
    device: R::Device,
    problem: &GemmProblem,
    strategy: &Strategy,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let flops = 2.0 * problem.b as f64 * problem.m as f64 * problem.n as f64 * problem.k as f64;

    let bench = GemmBench::<R> {
        b: problem.b,
        m: problem.m,
        n: problem.n,
        k: problem.k,
        lhs_layout: problem.lhs_layout,
        rhs_layout: problem.rhs_layout,
        strategy: strategy.clone(),
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
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
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
        let tl = matches!(self.lhs_layout, MatrixLayout::ColMajor);
        let tr = matches!(self.rhs_layout, MatrixLayout::ColMajor);

        let mut lhs = TensorHandle::empty(
            &client,
            vec![self.b, self.m, self.k],
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
            vec![self.b, self.k, self.n],
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
            vec![self.b, self.m, self.n],
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
