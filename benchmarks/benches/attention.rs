#![allow(
    unused,
    clippy::extra_unused_type_parameters,
    clippy::single_element_loop
)]

use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration},
    client::ComputeClient,
    future,
    prelude::*,
    profile::TimingMethod,
    std::tensor::TensorHandle,
};
use cubek::{
    attention::{
        self,
        definition::{
            AttentionDims, AttentionElems, AttentionGlobalTypes, AttentionIdent, AttentionOptions,
            AttentionPrecision, AttentionProblem, attention_types::*,
        },
        launch::{BlueprintStrategy, Strategy},
        routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
    },
    random::random_uniform,
};
use std::{default, marker::PhantomData};

pub struct AttentionInputs<R: Runtime> {
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

#[allow(dead_code)]
pub struct AttentionBench<R: Runtime, AP> {
    problem: AttentionProblem,
    strategy: Strategy,
    device: R::Device,
    client: ComputeClient<R>,
    _phantom: PhantomData<AP>,
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
        .map_err(|it| format!("{it:?}"))
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
            .map_err(|it| format!("{it:?}"))
    }
}

#[allow(dead_code)]
fn run<R: Runtime, AP: AttentionPrecision>(device: R::Device) {
    let client = R::client(&device);

    let global_dtypes = AttentionGlobalTypes::from_single_float_dtype(
        half::f16::as_type_native_unchecked(),
        AttentionGlobalTypes::mask_dtype(&client),
    );

    let bert = AttentionProblem {
        dims: AttentionDims {
            batch: 8,
            num_heads: 12,
            seq_q: 128,
            seq_kv: 128,
            head_dim: 64,
            val_dim: 64,
        },
        global_dtypes: global_dtypes.clone(),
        masked: false,
        options: Default::default(),
        address_type: Default::default(),
    };

    let gpt2 = AttentionProblem {
        dims: AttentionDims {
            batch: 4,
            num_heads: 12,
            seq_q: 1024,
            seq_kv: 1024,
            head_dim: 64,
            val_dim: 64,
        },
        global_dtypes: global_dtypes.clone(),
        masked: true,
        options: AttentionOptions {
            causal: true,
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    };

    let llama = AttentionProblem {
        dims: AttentionDims {
            batch: 4,
            num_heads: 32,
            seq_q: 2048,
            seq_kv: 2048,
            head_dim: 128,
            val_dim: 128,
        },
        global_dtypes: global_dtypes.clone(),
        masked: true,
        options: AttentionOptions {
            causal: true,
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    };

    let long_context = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 16,
            seq_q: 4096,
            seq_kv: 4096,
            head_dim: 128,
            val_dim: 128,
        },
        global_dtypes: global_dtypes.clone(),
        masked: true,
        options: AttentionOptions {
            causal: true,
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    };

    let encoder_decoder = AttentionProblem {
        dims: AttentionDims {
            batch: 2,
            num_heads: 16,
            seq_q: 512,
            seq_kv: 1024,
            head_dim: 128,
            val_dim: 128,
        },
        global_dtypes: global_dtypes.clone(),
        masked: false,
        options: AttentionOptions {
            causal: false,
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    };

    let my_bench = AttentionProblem {
        dims: AttentionDims {
            batch: 1,
            num_heads: 4,
            seq_q: 4096,
            seq_kv: 4096,
            head_dim: 64,
            val_dim: 64,
        },
        global_dtypes: global_dtypes.clone(),
        masked: true,
        options: AttentionOptions {
            causal: true,
            accumulator_precision: Default::default(),
        },
        address_type: Default::default(),
    };

    // for problem in [bert, gpt2, llama, long_context, encoder_decoder] {
    for problem in [my_bench] {
        for strategy in [Strategy::Unit(BlueprintStrategy::Inferred(()))] {
            let bench = AttentionBench::<R, AP> {
                problem: problem.clone(),
                strategy,
                client: client.clone(),
                device: device.clone(),
                _phantom: PhantomData,
            };

            println!("problem: {:?}", bench.problem);
            println!("{}", bench.name());
            println!("{}", bench.run(TimingMethod::System).unwrap());
        }
    }
}

#[allow(unused)]
fn run_benches<R: Runtime, AP: AttentionPrecision>() {
    let client = R::client(&Default::default());

    run::<R, AP>(Default::default());
}

fn main() {
    run_benches::<cubecl::TestRuntime, half::f16>();
}
