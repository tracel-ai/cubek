use super::{AttentionBackwardSpec, AttentionBackwardStrategy};
use crate::kernels::{
    CubeStagedAttentionBackwardRoutine, PlaneFragmentAttentionBackwardRoutine,
    UnitAttentionBackwardRoutine,
};
use crate::{
    AttentionBackwardProblem, AttentionBackwardRoutine, AttentionBackwardTensors,
    attention_backward,
};
use cubecl::benchmark::{Benchmark, ProfileDuration, TimingMethod};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubecl::{Device, future};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;
use cubek_test_utils::{RunSamples, TestInput};

/// `spec`'s backward pass in `f16`, run by `strategy` on `cubecl::TestRuntime`: the raw samples.
pub fn bench(
    strategy: &AttentionBackwardStrategy,
    spec: &AttentionBackwardSpec,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let bench = AttentionBackwardBench {
        spec: *spec,
        strategy: *strategy,
        client: device.client(),
        device,
        samples: num_samples,
    };
    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::System))
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;
    Ok(RunSamples::new(durations))
}

struct AttentionBackwardBench {
    spec: AttentionBackwardSpec,
    strategy: AttentionBackwardStrategy,
    device: Device,
    client: Client,
    samples: usize,
}

/// What a backward pass reads: the forward's inputs, its output and logsumexp, and the loss's
/// gradient against the output. Their values are uniform: what a row times does not depend on
/// them.
#[derive(Clone)]
struct AttentionBackwardInputs {
    query: TensorHandle,
    key: TensorHandle,
    value: TensorHandle,
    out: TensorHandle,
    out_gradient: TensorHandle,
    logsumexp: TensorHandle,
}

impl AttentionBackwardBench {
    fn uniform(&self, shape: Vec<usize>, element: ElemType, seed: u64) -> TensorHandle {
        TestInput::builder(self.client.clone(), Shape::from(shape))
            .dtype(element)
            .uniform(seed, 0., 1.)
            .generate_without_host_data()
    }

    fn tensors(&self, input: AttentionBackwardInputs) -> Result<AttentionBackwardTensors, String> {
        let spec = self.spec;
        let plain = PlainTensor::of_handle;
        AttentionBackwardTensors::new(
            &self.client,
            plain(input.query),
            plain(input.key),
            plain(input.value),
            plain(input.out),
            plain(input.out_gradient),
            plain(input.logsumexp),
            (spec.head_dim as f32).sqrt().recip(),
            spec.visibility,
        )
        .map_err(|why| format!("{why:?}"))
    }

    /// `routine`'s lead setting on `tensors`.
    fn lead<R: AttentionBackwardRoutine>(
        &self,
        routine: R,
        tensors: AttentionBackwardTensors,
    ) -> Result<(), String> {
        let hw = Hardware::from_client(&self.client);
        let lead = routine.settings(&hw).swap_remove(0);
        routine
            .launch_on(&hw, tensors, &lead)
            .map(|_| ())
            .map_err(|why| format!("{lead} does not serve this problem: {why:?}"))
    }
}

impl Benchmark for AttentionBackwardBench {
    type Input = AttentionBackwardInputs;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let spec = self.spec;
        let element = half::f16::elem_type_native();
        let (b, h, kv) = (spec.batch, spec.heads, spec.kv_heads);
        let (q, s, d) = (spec.seq_q, spec.seq_kv, spec.head_dim);
        AttentionBackwardInputs {
            query: self.uniform(vec![b, h, q, d], element, 0),
            key: self.uniform(vec![b, kv, s, d], element, 1),
            value: self.uniform(vec![b, kv, s, d], element, 2),
            out: self.uniform(vec![b, h, q, d], element, 3),
            out_gradient: self.uniform(vec![b, h, q, d], element, 4),
            logsumexp: self.uniform(vec![b, h, q], AttentionBackwardProblem::SUM, 5),
        }
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let tensors = self.tensors(input)?;
        match self.strategy {
            AttentionBackwardStrategy::UnitLead => self.lead(UnitAttentionBackwardRoutine, tensors),
            AttentionBackwardStrategy::PlaneLead => {
                self.lead(PlaneFragmentAttentionBackwardRoutine, tensors)
            }
            AttentionBackwardStrategy::CubeLead => {
                self.lead(CubeStagedAttentionBackwardRoutine, tensors)
            }
            AttentionBackwardStrategy::Tuned => {
                attention_backward(&self.device, tensors);
                Ok(())
            }
        }
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "{}-attention-backward-f16-{:?}",
            self.client.name(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "attention-backward-bench", || {
            self.execute(args)
        })
    }
}
