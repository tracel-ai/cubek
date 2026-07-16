use std::marker::PhantomData;

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

use crate::ReduceStrategy;
use crate::components::instructions::ReduceOperationConfig;
use crate::eval::benchmarks::problem::{ReduceBenchKind, ReduceProblem};

pub fn bench(
    strategy: &ReduceStrategy,
    problem: &ReduceProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

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

struct ReduceBench<E> {
    shape: Vec<usize>,
    axis: usize,
    config: ReduceOperationConfig,
    kind: ReduceBenchKind,
    strategy: ReduceStrategy,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

impl<E: Float> Benchmark for ReduceBench<E> {
    /// `(input, values, indices)`. The index tensor is allocated for every kind so
    /// that allocation never lands inside the timed section, but only the
    /// two-launch and fused kinds write to it.
    type Input = (
        TensorHandle<TestRuntime>,
        TensorHandle<TestRuntime>,
        TensorHandle<TestRuntime>,
    );
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);
        let elem = E::as_type_native_unchecked();
        let storage = elem.storage_type();

        let input = TestInput::builder(client.clone(), Shape::from(self.shape.clone()))
            .dtype(storage)
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
        let indices = TensorHandle::empty(&client, shape_out, u32::as_type_native_unchecked());

        (input, out, indices)
    }

    fn execute(&self, (input, out, indices): Self::Input) -> Result<(), String> {
        let value_dtype = E::as_type_native_unchecked().storage_type();
        let index_dtype = u32::as_type_native_unchecked().storage_type();
        let acc_dtype = f32::as_type_native_unchecked().storage_type();

        let k = match self.config {
            ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) => k,
            _ => 1,
        };

        match self.kind {
            ReduceBenchKind::Single => {
                let output_dtype = match self.config {
                    ReduceOperationConfig::ArgMax
                    | ReduceOperationConfig::ArgMin
                    | ReduceOperationConfig::ArgTopK(_) => index_dtype,
                    _ => value_dtype,
                };
                crate::reduce::<TestRuntime>(
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
                crate::reduce::<TestRuntime>(
                    &self.client,
                    input.clone().binding(),
                    out.binding(),
                    self.axis,
                    self.strategy.clone(),
                    ReduceOperationConfig::TopK(k),
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: value_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
                crate::reduce::<TestRuntime>(
                    &self.client,
                    input.binding(),
                    indices.binding(),
                    self.axis,
                    self.strategy.clone(),
                    ReduceOperationConfig::ArgTopK(k),
                    crate::ReduceDtypes {
                        input: value_dtype,
                        output: index_dtype,
                        accumulation: acc_dtype,
                    },
                )
                .map_err(|err| format!("{err}"))?;
            }
            ReduceBenchKind::Fused => {
                crate::reduce_with_indices::<TestRuntime>(
                    &self.client,
                    input.binding(),
                    out.binding(),
                    indices.binding(),
                    self.axis,
                    self.strategy.clone(),
                    ReduceOperationConfig::TopK(k),
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
        self.client
            .profile(|| self.execute(args), "reduce-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}-{:?}-{:?}-{:?}",
            self.axis,
            E::as_type_native_unchecked(),
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
