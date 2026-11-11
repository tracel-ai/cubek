//! Runs a routine over the blocks of runs and cases, against the reference.

use super::case::{Case, EPSILON, F16, F32, Run, blocks};
use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_rms_norm::kernels::{RmsNormCall, RmsNormGate, RmsNormRoutine, RmsNormRoutineCandidate};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, TestInput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Declines {
    Never,
    AsItsMicrokernelStates,
}

/// What a result written at `output` from `input` may stray from the reference, relative.
fn tolerance(input: ElemType, output: ElemType, gated: bool) -> f64 {
    match output {
        F32 if input == F32 && !gated => 1e-5,
        F32 => 1e-4,
        F16 => 4e-3,
        _ => 2e-2,
    }
}

/// `routine` over every block: each at the settings it lists or its default one, at each run the
/// device supports, on each case, against the reference.
pub fn check_routine<M: Microkernel, R: RmsNormRoutine<M> + Copy>(routine: R, declines: Declines) {
    let client = cubecl::test_device().client();
    let candidates = RmsNormRoutineCandidate::list(routine, &Hardware::from_client(&client));
    let mut launched = 0;
    for block in blocks() {
        let settings = match block.every_setting {
            true => &candidates[..],
            false => &candidates[..1],
        };
        let runs = block
            .runs
            .iter()
            .filter(|run| runs_on(&client, run.element) && runs_on(&client, run.output_element));
        for run in runs {
            for candidate in settings {
                for case in &block.cases {
                    let call = call(&client, case, run);
                    let what = format!("{} {run:?} {case:?}", candidate.settings());
                    let out = match (candidate.launch(&call), declines) {
                        (Ok(out), _) => out,
                        (Err(why), Declines::AsItsMicrokernelStates)
                            if why == (Refusal::Declined { why: M::DECLINE }) =>
                        {
                            continue;
                        }
                        (Err(why), _) => panic!("{what}: {why:?}"),
                    };
                    launched += 1;
                    // No rows is no launch, and an empty buffer has nothing to read back.
                    if case.rows == 0 {
                        assert_eq!(out.shape().num_elements(), 0, "{what}");
                        continue;
                    }
                    let tolerance = tolerance(run.element, run.output_element, run_gated(run));
                    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                    for (index, want) in reference(case, run).into_iter().enumerate() {
                        let got = got.data.get_f32(index) as f64;
                        assert!(
                            (got - want).abs() <= tolerance * (1.0 + want.abs()),
                            "{what}, element {index}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
    }
    assert!(launched > 0, "the routine ran no case on this device");
}

fn runs_on(client: &Client, element: ElemType) -> bool {
    let usage = client.properties().features.type_usage(element);
    usage.is_superset(TypeUsage::Arithmetic | TypeUsage::Buffer)
}

fn run_gated(run: &Run) -> bool {
    run.order.is_some()
}

fn reference(case: &Case, run: &Run) -> Vec<f64> {
    case.reference(run.element, run.order)
}

/// The norm `run` states of `case`, its operands uploaded.
fn call(client: &Client, case: &Case, run: &Run) -> RmsNormCall {
    let (element, weight) = (run.element, case.weight());
    let shape = [case.rows, case.dim];
    RmsNormCall {
        client: client.clone(),
        input: tensor(client, case.input(), &shape, element),
        weight: tensor(client, weight.clone(), &[weight.len()], element),
        epsilon: EPSILON,
        output_element: run.output_element,
        gate: run.order.map(|order| RmsNormGate {
            tensor: tensor(client, case.gate(), &shape, element),
            order,
        }),
    }
}

/// `values` uploaded as `element`, row-major in `shape`.
fn tensor(client: &Client, values: Vec<f32>, shape: &[usize], element: ElemType) -> PlainTensor {
    let handle = TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(element)
        .custom(values)
        .generate_without_host_data();
    PlainTensor::of_handle(handle)
}
