use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::eval::problem::{AttentionSpec, build_problem};
use crate::forward::definition::{
    AttentionGlobalTypes, AttentionIdent, AttentionPrecision, AttentionProblem, attention_types::*,
};
use crate::forward::launch::{self, Strategy};

/// Run one (strategy, spec) pair on `cubecl::TestRuntime` with `f16`
/// precision and return the raw samples.
pub fn bench(
    strategy: &Strategy,
    spec: &AttentionSpec,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let global_dtypes = AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(&client),
    );
    let problem = build_problem(spec, global_dtypes);

    let bench = AttentionBench::<half::f16> {
        problem,
        strategy: strategy.clone(),
        client: client.clone(),
        device,
        samples: num_samples,
        _phantom: std::marker::PhantomData,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct AttentionBench<AP> {
    problem: AttentionProblem,
    strategy: Strategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _phantom: std::marker::PhantomData<AP>,
}

struct AttentionInputs {
    query: TensorHandle<TestRuntime>,
    key: TensorHandle<TestRuntime>,
    value: TensorHandle<TestRuntime>,
    mask: Option<TensorHandle<TestRuntime>>,
}

impl Clone for AttentionInputs {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
            mask: self.mask.clone(),
        }
    }
}

fn make_uniform<T: Numeric>(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 4],
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(T::as_type_native_unchecked().storage_type())
        .uniform(seed, 0., 1.)
        .generate_without_host_data()
}

impl<AP: AttentionPrecision> Benchmark for AttentionBench<AP> {
    type Input = AttentionInputs;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);

        let query = make_uniform::<QG<AP>>(&client, self.problem.shape(AttentionIdent::Query), 0);
        let key = make_uniform::<KG<AP>>(&client, self.problem.shape(AttentionIdent::Key), 1);
        let value = make_uniform::<VG<AP>>(&client, self.problem.shape(AttentionIdent::Value), 2);
        let mask = self
            .problem
            .masked
            .then(|| make_uniform::<MSK<AP>>(&client, self.problem.shape(AttentionIdent::Mask), 3));

        AttentionInputs {
            query,
            key,
            value,
            mask,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let out: TensorHandle<TestRuntime> = TensorHandle::empty(
            &client,
            self.problem.shape(AttentionIdent::Out),
            self.problem.global_dtypes.out,
        );
        launch::launch_ref(
            self.strategy.clone(),
            &self.client,
            input.query.binding(),
            input.key.binding(),
            input.value.binding(),
            None,
            out.binding(),
            &self.problem.global_dtypes,
            self.problem.options.clone(),
        )
        .map_err(|e| format!("{e:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = <TestRuntime as Runtime>::client(&self.device);
        format!(
            "{}-attention-{}-{}-{}-{}--{:?}",
            <TestRuntime as Runtime>::name(&client),
            QG::<AP>::as_type_native_unchecked(),
            KG::<AP>::as_type_native_unchecked(),
            VG::<AP>::as_type_native_unchecked(),
            OG::<AP>::as_type_native_unchecked(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "attention-bench")
            .map(|it| it.1)
            .map_err(|e| format!("{e:?}"))
    }
}
