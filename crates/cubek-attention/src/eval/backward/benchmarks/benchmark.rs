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

use super::strategy::BackwardStrategy;
use crate::backward::{
    BackwardConfig, flash_attention_backward, flash_attention_backward_dkdv,
    flash_attention_backward_dq, flash_attention_backward_prepass,
};
use crate::eval::problem::{AttentionSpec, build_problem};
use crate::forward::definition::{
    AttentionGlobalTypes, AttentionIdent, AttentionPrecision, AttentionProblem, attention_types::*,
};
use crate::forward::launch::{BlueprintStrategy, Strategy, launch_ref_with_lse};

/// Run one `(strategy, spec)` pair on `cubecl::TestRuntime` with `f16`
/// precision and return the raw samples.
pub fn bench(
    strategy: &BackwardStrategy,
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

    let bench = BackwardBench::<half::f16> {
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

struct BackwardBench<AP> {
    problem: AttentionProblem,
    strategy: BackwardStrategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _phantom: std::marker::PhantomData<AP>,
}

struct BackwardInputs {
    q: TensorHandle<TestRuntime>,
    k: TensorHandle<TestRuntime>,
    v: TensorHandle<TestRuntime>,
    o: TensorHandle<TestRuntime>,
    lse: TensorHandle<TestRuntime>,
    do_: TensorHandle<TestRuntime>,
    dq: TensorHandle<TestRuntime>,
    dk: TensorHandle<TestRuntime>,
    dv: TensorHandle<TestRuntime>,
    d: TensorHandle<TestRuntime>,
}

impl Clone for BackwardInputs {
    fn clone(&self) -> Self {
        Self {
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
            o: self.o.clone(),
            lse: self.lse.clone(),
            do_: self.do_.clone(),
            dq: self.dq.clone(),
            dk: self.dk.clone(),
            dv: self.dv.clone(),
            d: self.d.clone(),
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

fn make_zeros<T: Numeric>(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 4],
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(T::as_type_native_unchecked().storage_type())
        .zeros()
        .generate_without_host_data()
}

fn make_zeros_row(
    client: &ComputeClient<TestRuntime>,
    shape: [usize; 3],
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f32::as_type_native_unchecked().storage_type())
        .zeros()
        .generate_without_host_data()
}

impl<AP: AttentionPrecision> Benchmark for BackwardBench<AP> {
    type Input = BackwardInputs;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let row_shape = [
            self.problem.dims.batch,
            self.problem.dims.num_heads,
            self.problem.dims.seq_q,
        ];

        BackwardInputs {
            q: make_uniform::<QG<AP>>(&client, self.problem.shape(AttentionIdent::Query), 0),
            k: make_uniform::<KG<AP>>(&client, self.problem.shape(AttentionIdent::Key), 1),
            v: make_uniform::<VG<AP>>(&client, self.problem.shape(AttentionIdent::Value), 2),
            o: make_uniform::<OG<AP>>(&client, self.problem.shape(AttentionIdent::Out), 3),
            lse: make_zeros_row(&client, row_shape),
            do_: make_uniform::<OG<AP>>(&client, self.problem.shape(AttentionIdent::Out), 4),
            dq: make_zeros::<QG<AP>>(&client, self.problem.shape(AttentionIdent::Query)),
            dk: make_zeros::<KG<AP>>(&client, self.problem.shape(AttentionIdent::Key)),
            dv: make_zeros::<VG<AP>>(&client, self.problem.shape(AttentionIdent::Value)),
            d: make_zeros_row(&client, row_shape),
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let cfg = {
            let mut c = BackwardConfig::from_head_dim(self.problem.dims.head_dim);
            c.causal = self.problem.options.causal;
            c
        };
        match self.strategy {
            BackwardStrategy::Prepass => flash_attention_backward_prepass(
                &self.client,
                input.o.binding(),
                input.do_.binding(),
                input.d.binding(),
            ),
            BackwardStrategy::Dq => flash_attention_backward_dq(
                &self.client,
                input.q.binding(),
                input.k.binding(),
                input.v.binding(),
                input.do_.binding(),
                input.lse.binding(),
                input.d.binding(),
                input.dq.binding(),
                &self.problem.global_dtypes,
                cfg,
            ),
            BackwardStrategy::Dkdv => flash_attention_backward_dkdv(
                &self.client,
                input.q.binding(),
                input.k.binding(),
                input.v.binding(),
                input.do_.binding(),
                input.lse.binding(),
                input.d.binding(),
                input.dk.binding(),
                input.dv.binding(),
                &self.problem.global_dtypes,
                cfg,
            ),
            BackwardStrategy::EndToEnd => flash_attention_backward(
                &self.client,
                input.q.binding(),
                input.k.binding(),
                input.v.binding(),
                input.o.binding(),
                input.lse.binding(),
                input.do_.binding(),
                input.dq.binding(),
                input.dk.binding(),
                input.dv.binding(),
                &self.problem.global_dtypes,
                cfg,
            ),
            BackwardStrategy::ForwardVsBackward => {
                // Forward first (with lse emission), then the full backward.
                // For now we record the combined time; the ratio plot can
                // split the two when the kernels start producing real
                // numbers. Both legs are `todo!()` stubs.
                launch_ref_with_lse(
                    Strategy::Unit(BlueprintStrategy::Inferred(())),
                    &self.client,
                    input.q.clone().binding(),
                    input.k.clone().binding(),
                    input.v.clone().binding(),
                    None,
                    input.o.clone().binding(),
                    Some(input.lse.clone().binding()),
                    &self.problem.global_dtypes,
                    self.problem.options.clone(),
                )
                .map_err(|e| format!("{e:?}"))?;
                flash_attention_backward(
                    &self.client,
                    input.q.binding(),
                    input.k.binding(),
                    input.v.binding(),
                    input.o.binding(),
                    input.lse.binding(),
                    input.do_.binding(),
                    input.dq.binding(),
                    input.dk.binding(),
                    input.dv.binding(),
                    &self.problem.global_dtypes,
                    cfg,
                )
            }
        }
        .map_err(|e| format!("{e:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = <TestRuntime as Runtime>::client(&self.device);
        format!(
            "{}-attention-backward-{:?}-{}",
            <TestRuntime as Runtime>::name(&client),
            self.strategy,
            QG::<AP>::as_type_native_unchecked(),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "attention-backward-bench")
            .map(|it| it.1)
            .map_err(|e| format!("{e:?}"))
    }
}
