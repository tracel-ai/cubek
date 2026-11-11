//! Runs a routine over the blocks of runs and cases, against the reference.

use super::case::{BF16, Case, EPSILON, F16, F32, Run, Slices, blocks};
use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::rms_norm::GateOrder;
use cubek_rms_norm::kernels::{
    RmsNormCall, RmsNormGate, RmsNormRoutine, RmsNormRoutineCandidate, RmsNormTensors,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType};

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
    let operand = |values: Vec<f32>, start: fn(Slices) -> usize| match case.slices {
        Some(slices) => view(
            client,
            &values,
            &[case.rows, case.dim],
            (slices.pitch, start(slices)),
            element,
        ),
        None => view(
            client,
            &values,
            &[case.rows, case.dim],
            (case.dim, 0),
            element,
        ),
    };
    let call = RmsNormCall {
        client: client.clone(),
        input: operand(case.input(), |slices| slices.input_start),
        weight: view(client, &weight, &[weight.len()], (weight.len(), 0), element),
        epsilon: EPSILON,
        output_element: run.output_element,
        gate: None,
    };
    let call = match case.group_size {
        Some(group_size) => call.grouped(group_size),
        None => call,
    };
    match run.order {
        Some(order) => call.gated(operand(case.gate(), |slices| slices.gate_start), order),
        None => call,
    }
}

/// A value no case holds, which a read from the wrong place would show.
const POISON: f32 = 123.0;

/// `values` as `element` holds them, in rows `pitch` elements apart, starting `start` elements
/// into a buffer that is `POISON` elsewhere.
fn view(
    client: &Client,
    values: &[f32],
    shape: &[usize],
    (pitch, start): (usize, usize),
    element: ElemType,
) -> PlainTensor {
    let dim = shape[shape.len() - 1];
    let rows = values.len() / dim;
    let mut buffer = vec![POISON; start + rows * pitch];
    for row in 0..rows {
        let at = start + row * pitch;
        buffer[at..at + dim].copy_from_slice(&values[row * dim..(row + 1) * dim]);
    }
    let bytes: Vec<u8> = match element {
        F16 => buffer
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect(),
        BF16 => buffer
            .iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect(),
        _ => buffer.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let handle = client
        .create_from_slice(&bytes)
        .offset_start((start * element.size()) as u64);
    let strides = if shape.len() == 1 {
        vec![1]
    } else {
        vec![pitch, 1]
    };
    PlainTensor::of_handle(TensorHandle::from_metadata(
        handle,
        Metadata::new(shape.to_vec(), strides),
        element,
    ))
}

/// A gate that is an aligned slice of a wider projection, beside a dense input, is read where it
/// lies: its view keeps a start past its buffer's own.
pub fn assert_gate_slice_is_read_in_place() {
    let client = cubecl::test_device().client();
    if !runs_on(&client, F16) {
        return;
    }
    let (rows, dim) = (5, 1024);
    let case = Case::new(rows, dim);
    let weight = case.weight();
    let call = RmsNormCall {
        client: client.clone(),
        input: view(&client, &case.input(), &[rows, dim], (dim, 0), F16),
        weight: view(&client, &weight, &[dim], (dim, 0), F16),
        epsilon: EPSILON,
        output_element: F16,
        gate: Some(RmsNormGate {
            tensor: view(&client, &case.gate(), &[rows, dim], (3 * dim, dim), F16),
            order: GateOrder::GateThenNorm,
        }),
    }
    .grouped(128);
    let tensors = RmsNormTensors::new(&call).expect("the device runs the norm");
    assert_ne!(tensors.problem().start_bytes.gate, 0, "the gate was copied");
}
