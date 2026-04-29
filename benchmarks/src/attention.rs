//! Attention benchmark registry. Strategy/problem IDs are stable strings; the
//! tuner UI uses them to drive runs across cubek versions.

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
            AttentionDims, AttentionGlobalTypes, AttentionIdent, AttentionOptions,
            AttentionPrecision, AttentionProblem, attention_types::*,
        },
        launch::{BlueprintStrategy, Strategy},
        routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
    },
    random::random_uniform,
};

use crate::registry::{ItemDescriptor, RunSamples};

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_UNIT: &str = "unit_inferred";
pub const STRATEGY_BLACKBOX_ACCELERATED: &str = "blackbox_accelerated_inferred";

pub const PROBLEM_BERT: &str = "bert";
pub const PROBLEM_GPT2: &str = "gpt2";
pub const PROBLEM_LLAMA: &str = "llama";
pub const PROBLEM_LONG_CONTEXT: &str = "long_context";
pub const PROBLEM_ENCODER_DECODER: &str = "encoder_decoder";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_UNIT,
            label: "Unit (inferred)",
        },
        ItemDescriptor {
            id: STRATEGY_BLACKBOX_ACCELERATED,
            label: "Blackbox accelerated (inferred, np=1 sq=1 skv=1)",
        },
    ]
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_BERT,
            label: "BERT (b=8 h=12 sq=skv=128 d=64)",
        },
        ItemDescriptor {
            id: PROBLEM_GPT2,
            label: "GPT-2 (b=4 h=12 sq=skv=1024 d=64, causal+mask)",
        },
        ItemDescriptor {
            id: PROBLEM_LLAMA,
            label: "Llama (b=4 h=32 sq=skv=2048 d=128, causal+mask)",
        },
        ItemDescriptor {
            id: PROBLEM_LONG_CONTEXT,
            label: "Long context (b=1 h=16 sq=skv=4096 d=128, causal+mask)",
        },
        ItemDescriptor {
            id: PROBLEM_ENCODER_DECODER,
            label: "Encoder-decoder (b=2 h=16 sq=512 skv=1024 d=128)",
        },
    ]
}

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

    Ok(RunSamples {
        durations,
        tflops: None,
    })
}

fn strategy_for(id: &str) -> Option<Strategy> {
    match id {
        STRATEGY_UNIT => Some(Strategy::Unit(BlueprintStrategy::Inferred(()))),
        STRATEGY_BLACKBOX_ACCELERATED => Some(Strategy::BlackboxAccelerated(
            BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
                num_planes: 1,
                seq_q: 1,
                seq_kv: 1,
            }),
        )),
        _ => None,
    }
}

fn problem_for(id: &str, global_dtypes: AttentionGlobalTypes) -> Option<AttentionProblem> {
    let causal_masked = AttentionOptions {
        causal: true,
        accumulator_precision: Default::default(),
    };
    Some(match id {
        PROBLEM_BERT => AttentionProblem {
            dims: AttentionDims {
                batch: 8,
                num_heads: 12,
                seq_q: 128,
                seq_kv: 128,
                head_dim: 64,
                val_dim: 64,
            },
            global_dtypes,
            masked: false,
            options: Default::default(),
            address_type: Default::default(),
        },
        PROBLEM_GPT2 => AttentionProblem {
            dims: AttentionDims {
                batch: 4,
                num_heads: 12,
                seq_q: 1024,
                seq_kv: 1024,
                head_dim: 64,
                val_dim: 64,
            },
            global_dtypes,
            masked: true,
            options: causal_masked,
            address_type: Default::default(),
        },
        PROBLEM_LLAMA => AttentionProblem {
            dims: AttentionDims {
                batch: 4,
                num_heads: 32,
                seq_q: 2048,
                seq_kv: 2048,
                head_dim: 128,
                val_dim: 128,
            },
            global_dtypes,
            masked: true,
            options: causal_masked,
            address_type: Default::default(),
        },
        PROBLEM_LONG_CONTEXT => AttentionProblem {
            dims: AttentionDims {
                batch: 1,
                num_heads: 16,
                seq_q: 4096,
                seq_kv: 4096,
                head_dim: 128,
                val_dim: 128,
            },
            global_dtypes,
            masked: true,
            options: causal_masked,
            address_type: Default::default(),
        },
        PROBLEM_ENCODER_DECODER => AttentionProblem {
            dims: AttentionDims {
                batch: 2,
                num_heads: 16,
                seq_q: 512,
                seq_kv: 1024,
                head_dim: 128,
                val_dim: 128,
            },
            global_dtypes,
            masked: false,
            options: AttentionOptions {
                causal: false,
                accumulator_precision: Default::default(),
            },
            address_type: Default::default(),
        },
        _ => return None,
    })
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
