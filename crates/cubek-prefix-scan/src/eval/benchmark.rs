use crate::{
    eval::{
        correctness::ScanCorrectness,
        problem::{ScanProblem, problems},
        strategy::{ScanStrategy, strategies},
    },
    prefix_scan_launch,
};
use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    client::Client,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, TestInput};

struct ScanBench {
    problem: ScanProblem,
    strategy: ScanStrategy,
    device: cubecl::Device,
    client: Client,
    dtype: ElemType,
    samples: usize,
}

impl Benchmark for ScanBench {
    type Input = TensorHandle;
    type Output = TensorHandle;

    fn prepare(&self) -> Self::Input {
        TestInput::builder(self.client.clone(), self.problem.shape.clone())
            .dtype(self.dtype)
            .uniform(0, -1., 1.)
            .generate_without_host_data()
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle, String> {
        let output = TensorHandle::empty(&self.client, input.shape().to_vec(), self.dtype);

        prefix_scan_launch(
            &self.client,
            input.binding(),
            output.clone().binding(),
            self.dtype,
            self.problem.scan_kind,
            self.problem.scan_instruction_kind,
            self.strategy.clone(),
        )
        .map_err(|err| format!("{err}"))?;

        Ok(output)
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "prefix-scan-{}-{}-{:?}-{:?}-{:?}-{:?}",
            self.problem.scan_kind.label(),
            self.problem.scan_instruction_kind.label(),
            self.problem.shape,
            self.strategy,
            self.dtype,
            self.device,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = ScanProblem;
    type Strategy = ScanStrategy;

    fn id(&self) -> &'static str {
        "prefix_scan"
    }

    fn label(&self) -> &'static str {
        "Prefix Scan"
    }

    fn problems(&self) -> Vec<CatalogEntry<ScanProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<ScanStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &ScanStrategy,
        problem: &ScanProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        let device = cubecl::test_device();
        let client = device.client();
        let dtype = f32::elem_type_native();

        let bench = ScanBench {
            problem: problem.clone(),
            strategy: strategy.clone(),
            device,
            client,
            dtype,
            samples: num_samples,
        };

        let durations = bench
            .run(cubek_test_utils::timing_method(TimingMethod::Device))
            .map_err(|e| format!("benchmark failed: {e}"))?
            .durations;

        Ok(RunSamples::new(durations))
    }

    fn timing_method(&self) -> TimingMethod {
        cubek_test_utils::timing_method(TimingMethod::Device)
    }

    fn correctness(
        &self,
    ) -> Option<&dyn cubek_test_utils::Correctness<Problem = ScanProblem, Strategy = ScanStrategy>>
    {
        Some(&ScanCorrectness)
    }

    fn work(&self, problem: &ScanProblem) -> Option<CategoryWork> {
        let dtype = f32::elem_type_native();
        let num_elements = problem.shape.iter().product::<usize>();
        Some(CategoryWork {
            compute: Some(ComputeWork::direct(num_elements * 5, dtype)),
            bytes_read: num_elements * dtype.size(),
            bytes_written: num_elements * dtype.size(),
        })
    }
}
