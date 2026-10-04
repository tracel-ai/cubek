use crate::{
    base::{PrefixScanSettings, PrefixScanStrategy},
    eval::{cpu_reference::cpu_reference_prefix_scan, tune::PrefixScanTuneKey},
    launch::prefix_scan_launch,
};
use cubecl::{client::Client, std::tensor::TensorHandle, zspace::Strides};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataVec, Progress, TestInput, launch_and_capture_outcome,
};

pub fn device_result(
    client: Client,
    strategy: PrefixScanStrategy,
    problem: PrefixScanTuneKey,
    seed: u64,
) -> Result<HostData, String> {
    let (input, _input_data) = TestInput::builder(client.clone(), &problem.shape)
        .uniform(seed, -1., 1.)
        .generate_with_host_data(problem.dtype.into());
    let output = TensorHandle::empty(&client, &problem.shape, problem.dtype);
    let settings = PrefixScanSettings::new(client.properties(), problem.dtype, strategy);
    let outcome = launch_and_capture_outcome(&client, &[&output.handle], |client| {
        prefix_scan_launch(
            client,
            input.binding(),
            output.clone().binding(),
            problem.dtype,
            problem.scan_kind,
            problem.scan_instruction_kind,
            &settings,
        )
        .into()
    });
    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            output,
            problem.dtype.into(),
        )),
    }
}

pub fn cpu_reference_result(
    client: Client,
    problem: PrefixScanTuneKey,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    if let Some(p) = progress {
        p.set_total(problem.shape.iter().product::<usize>() as u64);
    }
    let (_input, input_data) = TestInput::builder(client.clone(), &problem.shape)
        .uniform(seed, -1., 1.)
        .generate_with_host_data(problem.dtype.into());
    let output_data = cpu_reference_prefix_scan(
        &input_data,
        problem.scan_kind,
        problem.scan_instruction_kind,
    );
    Ok(HostData {
        data: HostDataVec::F32(output_data),
        shape: problem.shape,
        strides: Strides::new(&[1]),
    })
}

pub struct PrefixScanCorrectness;

impl cubek_test_utils::Correctness for PrefixScanCorrectness {
    type Problem = PrefixScanTuneKey;
    type Strategy = PrefixScanStrategy;

    fn kernel_result(
        &self,
        strategy: &PrefixScanStrategy,
        problem: &PrefixScanTuneKey,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = cubecl::test_device();
        let client = device.client();
        device_result(client, strategy.clone(), problem.clone(), seeds[0])
    }

    fn reference_result(
        &self,
        problem: &PrefixScanTuneKey,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = cubecl::test_device();
        let client = device.client();
        cpu_reference_result(client, problem.clone(), seeds[0], progress)
    }
}
