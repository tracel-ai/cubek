use std::marker::PhantomData;

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use crate::ReduceStrategy;
use crate::components::instructions::ReduceOperationConfig;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};

pub fn bench(
    strategy: &ReduceStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let bench = ReduceBench::<f32> {
        shape: problem.shape.clone(),
        axis: problem.axis,
        config: problem.config,
        kind: problem.kind,
        strategy: strategy.clone(),
        device,
        client,
        samples: num_samples,
        _e: PhantomData,
    };

    // Device timing (hardware timestamps) rather than system timing: the reduce
    // problems are ~270 MB, and wall-clock timing of launch+sync picked up enough
    // host-side noise that identical kernels varied by over 10x between runs,
    // which made fused-vs-two-launch comparisons meaningless.
    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

/// The pair of single-output configs a caller runs today to get both halves of
/// `config`.
fn two_launch_configs(
    config: ReduceOperationConfig,
) -> (ReduceOperationConfig, ReduceOperationConfig) {
    match config {
        ReduceOperationConfig::TopK(k) | ReduceOperationConfig::ArgTopK(k) => (
            ReduceOperationConfig::TopK(k),
            ReduceOperationConfig::ArgTopK(k),
        ),
        ReduceOperationConfig::Max | ReduceOperationConfig::ArgMax => {
            (ReduceOperationConfig::Max, ReduceOperationConfig::ArgMax)
        }
        ReduceOperationConfig::Min | ReduceOperationConfig::ArgMin => {
            (ReduceOperationConfig::Min, ReduceOperationConfig::ArgMin)
        }
        other => panic!("{other:?} has no values/indices pair to compare"),
    }
}

struct ReduceBench<E> {
    shape: Vec<usize>,
    axis: usize,
    config: ReduceOperationConfig,
    kind: ReduceBenchKind,
    strategy: ReduceStrategy,
    device: cubecl::Device,
    client: Client,
    samples: usize,
    _e: PhantomData<E>,
}

impl<E: Float> Benchmark for ReduceBench<E> {
    /// `(input, values, indices)`. The index tensor is allocated for every kind so
    /// that allocation never lands inside the timed section, but only the
    /// two-launch and fused kinds write to it.
    type Input = (TensorHandle, TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = self.device.client();
        let elem = E::elem_type_native();

        let input = TestInput::builder(client.clone(), Shape::from(self.shape.clone()))
            .dtype(elem)
            .uniform(0, 0., 1.)
            .generate_without_host_data();
        let mut shape_out = self.shape.clone();
        let reduce_len = match self.config {
            ReduceOperationConfig::ArgTopK(len) => len,
            ReduceOperationConfig::TopK(len) => len,
            _ => 1,
        };
        shape_out[self.axis] = reduce_len;
        let out = TensorHandle::empty(&client, shape_out.clone(), elem);
        let indices = TensorHandle::empty(&client, shape_out, u32::elem_type_native());

        (input, out, indices)
    }

    fn execute(&self, (input, out, indices): Self::Input) -> Result<(), String> {
        let value_dtype = E::elem_type_native();
        let index_dtype = u32::elem_type_native();
        let acc_dtype = f32::elem_type_native();

        match self.kind {
            ReduceBenchKind::Single => {
                let output_dtype = match self.config {
                    ReduceOperationConfig::ArgMax
                    | ReduceOperationConfig::ArgMin
                    | ReduceOperationConfig::ArgTopK(_) => index_dtype,
                    _ => value_dtype,
                };
                crate::reduce(
                    &self.client,
                    input.binding(),
                    out.binding(),
                    self.axis,
                    self.strategy.clone(),
                    self.config,
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: output_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
            // What a caller needing both halves does today: run the whole
            // reduction twice, discarding half of each result.
            ReduceBenchKind::TwoLaunch => {
                let (values_config, indices_config) = two_launch_configs(self.config);
                crate::reduce(
                    &self.client,
                    input.clone().binding(),
                    out.binding(),
                    self.axis,
                    self.strategy.clone(),
                    values_config,
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: value_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
                crate::reduce(
                    &self.client,
                    input.binding(),
                    indices.binding(),
                    self.axis,
                    self.strategy.clone(),
                    indices_config,
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: index_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
            ReduceBenchKind::Fused => {
                crate::reduce_with_indices(
                    &self.client,
                    input.binding(),
                    out.binding(),
                    indices.binding(),
                    self.axis,
                    self.strategy.clone(),
                    self.config,
                    crate::ReduceWithIndicesDtypes {
                        input: value_dtype,
                        values: value_dtype,
                        indices: index_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
        }

        Ok(())
    }

    /// Measure with device timestamps around the launch, so the reported duration
    /// is the kernel's, not the host's view of launch+sync.
    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        let (launched, duration) = self
            .client
            .profile(|| self.execute(args), "reduce-bench")
            .map_err(|err| format!("{err:?}"))?;
        launched.map(|_| duration)
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}-{:?}-{:?}-{:?}",
            self.axis,
            E::elem_type_native(),
            self.shape,
            self.strategy,
            self.config,
            self.kind,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
