use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{RunSamples, TestInput};

use crate::definition::{MatmulElems, MatmulPrecision};
use crate::eval::benchmarks::gemm::problem::{GemmProblem, Precision};
use crate::{launch::launch_ref, strategy::Strategy};

pub fn bench(
    strategy: &Strategy,
    problem: &GemmProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    match problem.precision {
        Precision::F32 => bench_with::<f32>(problem, strategy, num_samples),
        Precision::F16 => bench_with::<half::f16>(problem, strategy, num_samples),
    }
}

fn bench_with<MP: MatmulPrecision>(
    problem: &GemmProblem,
    strategy: &Strategy,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let flops = 2.0 * problem.b as f64 * problem.m as f64 * problem.n as f64 * problem.k as f64;

    let bench = GemmBench {
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

struct GemmBench {
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    strategy: Strategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    dtypes: MatmulElems,
    samples: usize,
}

fn make_uniform(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 3],
    dtype: StorageType,
    seed: u64,
    lo: f32,
    hi: f32,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(dtype)
        .uniform(seed, lo, hi)
        .generate_without_host_data()
}

impl Benchmark for GemmBench {
    type Input = (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let tl = matches!(self.lhs_layout, MatrixLayout::ColMajor);
        let tr = matches!(self.rhs_layout, MatrixLayout::ColMajor);

        let mut lhs = make_uniform(
            &client,
            [self.b, self.m, self.k],
            self.dtypes.lhs_global,
            0,
            0.0,
            1.0,
        );
        if tl {
            let len = lhs.metadata.rank();
            lhs.metadata.strides_mut().swap(len - 2, len - 1);
        }

        let mut rhs = make_uniform(
            &client,
            [self.b, self.k, self.n],
            self.dtypes.rhs_global,
            1,
            0.0,
            1.1,
        );
        if tr {
            let len = rhs.metadata.rank();
            rhs.metadata.strides_mut().swap(len - 2, len - 1);
        }

        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let client = <TestRuntime as Runtime>::client(&self.device);
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
        let client = <TestRuntime as Runtime>::client(&self.device);
        format!(
            "{}-matmul-Lhs<{}-{}-{}>-Rhs<{}-{}-{}>-{}-{}-{}",
            <TestRuntime as Runtime>::name(&client),
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
