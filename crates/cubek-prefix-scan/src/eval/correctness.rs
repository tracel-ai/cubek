use crate::{
    eval::{
        cpu_reference::cpu_reference_prefix_scan, problem::ScanProblem, strategy::ScanStrategy,
    },
    prefix_scan_launch,
};
use cubecl::{client::Client, frontend::Scalar, std::tensor::TensorHandle, zspace::Strides};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataVec, Progress, TestInput, launch_and_capture_outcome,
};

pub fn device_result(
    client: Client,
    strategy: ScanStrategy,
    problem: ScanProblem,
    _seed: u64,
) -> Result<HostData, String> {
    let dtype = f32::elem_type_native();
    let (input, _input_data) = TestInput::builder(client.clone(), &problem.shape)
        .dtype(dtype)
        .custom(vec![1f32; problem.shape.iter().product()])
        .generate_with_host_data(dtype.into());
    let output = TensorHandle::empty(&client, &problem.shape, dtype);
    let outcome = launch_and_capture_outcome(&client, &[&output.handle], |client| {
        prefix_scan_launch(
            client,
            input.binding(),
            output.clone().binding(),
            dtype,
            problem.scan_kind,
            problem.scan_instruction_kind,
            strategy,
        )
        .into()
    });
    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => {
            Ok(HostData::from_tensor_handle(&client, output, dtype.into()))
        }
    }
}

pub fn cpu_reference_result(
    client: Client,
    problem: ScanProblem,
    _seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    if let Some(p) = progress {
        p.set_total(problem.shape.iter().product::<usize>() as u64);
    }
    let dtype = f32::elem_type_native();
    let (_input, input_data) = TestInput::builder(client.clone(), &problem.shape)
        .dtype(dtype)
        .custom(vec![1f32; problem.shape.iter().product()])
        .generate_with_host_data(dtype.into());
    /*let (_input, input_data) = TestInput::builder(client.clone(), &[problem.size])
    .uniform(seed, -1., 1.)
    .generate_with_f32_host_data();*/
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

pub struct ScanCorrectness;

impl cubek_test_utils::Correctness for ScanCorrectness {
    type Problem = ScanProblem;
    type Strategy = ScanStrategy;

    fn kernel_result(
        &self,
        strategy: &ScanStrategy,
        problem: &ScanProblem,
        seeds: &[u64],
    ) -> Result<HostData, String> {
        let device = cubecl::test_device();
        let client = device.client();
        device_result(client, strategy.clone(), problem.clone(), seeds[0])
    }

    fn reference_result(
        &self,
        problem: &ScanProblem,
        seeds: &[u64],
        progress: Option<&Progress>,
    ) -> Result<HostData, String> {
        let device = cubecl::test_device();
        let client = device.client();
        cpu_reference_result(client, problem.clone(), seeds[0], progress)
    }
}
