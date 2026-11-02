use super::{GatedDeltaWalkSpec, GatedDeltaWalkStrategy};
use crate::kernels::walk::{
    CubeGatedDeltaWalkRoutine, GatedDeltaWalkInputs, GatedDeltaWalkRoutine, launch_tuned, serves,
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

/// `spec`'s walk in `f32`, run by `strategy` on `cubecl::TestRuntime`: the raw samples.
pub fn bench(
    strategy: &GatedDeltaWalkStrategy,
    spec: &GatedDeltaWalkSpec,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let bench = GatedDeltaWalkBench {
        spec: *spec,
        strategy: *strategy,
        client: device.client(),
        device,
        samples: num_samples,
    };
    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::Device))
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;
    Ok(RunSamples::new(durations))
}

struct GatedDeltaWalkBench {
    spec: GatedDeltaWalkSpec,
    strategy: GatedDeltaWalkStrategy,
    device: Device,
    client: Client,
    samples: usize,
}

impl GatedDeltaWalkBench {
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

impl Benchmark for GatedDeltaWalkBench {
    type Input = GatedDeltaWalkInputs;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let GatedDeltaWalkSpec {
            batch,
            positions,
            heads,
        } = self.spec;
        let (b, t, kh, h) = (batch, positions, heads.key_heads, heads.value_heads());
        GatedDeltaWalkInputs {
            client: self.client.clone(),
            query: self.uniform(vec![b, t, kh, heads.key_features], 0, -0.09, 0.09),
            key: self.uniform(vec![b, t, kh, heads.key_features], 1, -0.09, 0.09),
            value: self.uniform(vec![b, t, h, heads.value_features], 2, -1.0, 1.0),
            decay: self.uniform(vec![b, t, h], 3, -1.0, -0.05),
            strength: self.uniform(vec![b, t, h], 4, 0.0, 1.0),
            state: self.uniform(
                vec![b, h, heads.key_features, heads.value_features],
                5,
                -0.2,
                0.2,
            ),
            output_element: f32::elem_type_native(),
            kept: NonZeroUsize::MIN,
        }
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        match self.strategy {
            GatedDeltaWalkStrategy::CubeLead => {
                let hw = Hardware::from_client(&self.client);
                CubeGatedDeltaWalkRoutine
                    .launch_lead_on(&hw, inputs)
                    .map(|_| ())
                    .map_err(|why| format!("the lead does not serve this problem: {why:?}"))
            }
            GatedDeltaWalkStrategy::Tuned => {
                let hw = Hardware::from_client(&self.client);
                serves(&hw, self.spec.heads, f32::elem_type_native())
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
            "{}-gated-delta-walk-f32-{:?}",
            self.client.name(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "gated-delta-walk-bench", || {
            self.execute(args)
        })
    }
}
