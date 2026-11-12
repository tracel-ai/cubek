use super::{RecurrenceWalkSpec, RecurrenceWalkStrategy};
use crate::kernels::walk::{
    CubeDeltaWalkRoutine, RecurrenceWalkInputs, RecurrenceWalkRoutine, RecurrenceWalkStateInputs,
    RecurrenceWalkStateLayout, launch_tuned, serves,
};
use core::num::NonZeroUsize;
use cubecl::benchmark::{Benchmark, ProfileDuration, TimingMethod};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubecl::{Device, future};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;
use cubek_test_utils::{RunSamples, TestInput};
use std::sync::atomic::{AtomicUsize, Ordering};

/// `spec`'s walk in `f32`, run by `strategy` on `cubecl::TestRuntime`: the raw samples.
pub fn bench(
    strategy: &RecurrenceWalkStrategy,
    spec: &RecurrenceWalkSpec,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();
    let bench = RecurrenceWalkBench {
        spec: *spec,
        strategy: *strategy,
        rotation: SlotRotation::new(&client, spec),
        client,
        device,
        samples: num_samples,
    };
    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::Device))
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;
    Ok(RunSamples::new(durations))
}

/// The slot tables a walk in a store's slots takes in turn: each call's rows lie in slots no call
/// since the store's last pass touched, so the states are read from memory and not from the cache.
struct SlotRotation {
    tables: Vec<PlainTensor>,
    next: AtomicUsize,
}

impl SlotRotation {
    /// The tables that place `spec`'s rows in consecutive slots, wrapping around the store; none
    /// for a walk over states it owns.
    fn new(client: &Client, spec: &RecurrenceWalkSpec) -> Self {
        let tables = match spec.state {
            RecurrenceWalkStateLayout::Owned { .. } => Vec::new(),
            RecurrenceWalkStateLayout::Slots { slots } => (0..slots.get() / spec.batch)
                .map(|turn| {
                    let rows: Vec<u32> = (0..spec.batch)
                        .map(|row| ((turn * spec.batch + row) % slots) as u32)
                        .collect();
                    PlainTensor::of_u32s(client, &[spec.batch], &rows)
                })
                .collect(),
        };
        Self {
            tables,
            next: AtomicUsize::new(0),
        }
    }

    /// `inputs` with the next table of the rotation placing its rows.
    fn turned(&self, mut inputs: RecurrenceWalkInputs) -> RecurrenceWalkInputs {
        if let RecurrenceWalkStateInputs::Slots { slots, .. } = &mut inputs.state {
            let turn = self.next.fetch_add(1, Ordering::Relaxed);
            *slots = self.tables[turn % self.tables.len()].clone();
        }
        inputs
    }
}

struct RecurrenceWalkBench {
    spec: RecurrenceWalkSpec,
    strategy: RecurrenceWalkStrategy,
    rotation: SlotRotation,
    device: Device,
    client: Client,
    samples: usize,
}

impl RecurrenceWalkBench {
    /// A tensor of values in `[low, high)`: what a row times does not depend on them, and the
    /// ranges keep the state bounded over a long walk.
    fn uniform(&self, shape: Vec<usize>, seed: u64, low: f32, high: f32) -> PlainTensor {
        PlainTensor::of_handle(
            TestInput::builder(self.client.clone(), Shape::from(shape))
                .dtype(f32::elem_type_native())
                .uniform(seed, low, high)
                .generate_without_host_data(),
        )
    }
}

impl Benchmark for RecurrenceWalkBench {
    type Input = RecurrenceWalkInputs;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let RecurrenceWalkSpec {
            batch,
            positions,
            heads,
            state,
        } = self.spec;
        let (b, t, kh, h) = (batch, positions, heads.key_heads, heads.value_heads());
        let state = match state {
            RecurrenceWalkStateLayout::Owned { .. } => RecurrenceWalkStateInputs::Owned {
                state: self.uniform(
                    vec![b, h, heads.key_features, heads.value_features],
                    5,
                    -0.2,
                    0.2,
                ),
                kept: NonZeroUsize::MIN,
            },
            RecurrenceWalkStateLayout::Slots { slots } => RecurrenceWalkStateInputs::Slots {
                store: self.uniform(
                    vec![slots.get(), h, heads.key_features, heads.value_features],
                    5,
                    -0.2,
                    0.2,
                ),
                slots: self.rotation.tables[0].clone(),
                lengths: PlainTensor::of_u32s(&self.client, &[b], &vec![1; b]),
            },
        };
        RecurrenceWalkInputs {
            client: self.client.clone(),
            query: self.uniform(vec![b, t, kh, heads.key_features], 0, -0.09, 0.09),
            key: self.uniform(vec![b, t, kh, heads.key_features], 1, -0.09, 0.09),
            value: self.uniform(vec![b, t, h, heads.value_features], 2, -1.0, 1.0),
            decay: self.uniform(vec![b, t, h], 3, -1.0, -0.05),
            strength: self.uniform(vec![b, t, h], 4, 0.0, 1.0),
            state,
            output_element: f32::elem_type_native(),
        }
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        let inputs = self.rotation.turned(inputs);
        match self.strategy {
            RecurrenceWalkStrategy::CubeLead => {
                let hw = Hardware::from_client(&self.client);
                CubeDeltaWalkRoutine
                    .launch_lead_on(&hw, inputs)
                    .map(|_| ())
                    .map_err(|why| format!("the lead does not serve this problem: {why:?}"))
            }
            RecurrenceWalkStrategy::Tuned => {
                let hw = Hardware::from_client(&self.client);
                serves(
                    &hw,
                    self.spec.heads,
                    self.spec.state.into(),
                    f32::elem_type_native(),
                )
                .map_err(|why| format!("the table does not serve this problem: {why:?}"))?;
                launch_tuned(&self.device, inputs, GroupPriority::DEFAULT);
                Ok(())
            }
        }
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "{}-recurrence-walk-f32-{:?}",
            self.client.name(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "recurrence-walk-bench", || {
            self.execute(args)
        })
    }
}
