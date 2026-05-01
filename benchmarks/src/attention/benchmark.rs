use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    attention::{
        self,
        definition::{
            AttentionGlobalTypes, AttentionIdent, AttentionPrecision, AttentionProblem,
            attention_types::*,
        },
        launch::Strategy,
    },
    random::random_uniform,
};

use crate::{
    attention::{problem::problem_for, strategy::strategy_for},
    registry::RunSamples,
};

/// Run one (strategy, problem) pair on `cubecl::TestRuntime` with `f16`
/// precision and return the raw samples.
pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime, half::f16>(
        Default::default(),
        strategy_id,
        problem_id,
        num_samples,
    )
}

pub fn run_on<R: Runtime, AP: AttentionPrecision>(
    device: R::Device,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let global_dtypes = AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(&client),
    );

    let problem = problem_for(problem_id, global_dtypes)
        .ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    let bench = AttentionBench::<R, AP> {
        problem,
        strategy,
        client: client.clone(),
        device,
        samples: num_samples,
        _phantom: PhantomData,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct AttentionBench<R: Runtime, AP> {
    problem: AttentionProblem,
    strategy: Strategy,
    device: R::Device,
    client: ComputeClient<R>,
    samples: usize,
    _phantom: PhantomData<AP>,
}

struct AttentionInputs<R: Runtime> {
    query: TensorHandle<R>,
    key: TensorHandle<R>,
    value: TensorHandle<R>,
    mask: Option<TensorHandle<R>>,
}

impl<R: Runtime> Clone for AttentionInputs<R> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
            mask: self.mask.clone(),
        }
    }
}

impl<R: Runtime, AP: AttentionPrecision> Benchmark for AttentionBench<R, AP> {
    type Input = AttentionInputs<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        fn make_random<R: Runtime, T: Numeric>(
            client: &ComputeClient<R>,
            shape: Vec<usize>,
        ) -> TensorHandle<R> {
            let dtype = T::as_type_native_unchecked();
            let tensor = TensorHandle::empty(client, shape, dtype);
            random_uniform(
                client,
                0.,
                1.,
                tensor.clone().binding(),
                dtype.storage_type(),
            )
            .unwrap();
            tensor
        }

        let query =
            make_random::<R, QG<AP>>(&client, self.problem.shape(AttentionIdent::Query).to_vec());
        let key =
            make_random::<R, KG<AP>>(&client, self.problem.shape(AttentionIdent::Key).to_vec());
        let value =
            make_random::<R, VG<AP>>(&client, self.problem.shape(AttentionIdent::Value).to_vec());
        let mask = self.problem.masked.then(|| {
            make_random::<R, MSK<AP>>(&client, self.problem.shape(AttentionIdent::Mask).to_vec())
        });

        AttentionInputs {
            query,
            key,
            value,
            mask,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let client = R::client(&self.device);
        let out: TensorHandle<R> = TensorHandle::empty(
            &client,
            self.problem.shape(AttentionIdent::Out).to_vec(),
            self.problem.global_dtypes.out,
        );
        attention::launch::launch_ref(
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
        let client = R::client(&self.device);
        format!(
            "{}-attention-{}-{}-{}-{}--{:?}",
            R::name(&client),
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
