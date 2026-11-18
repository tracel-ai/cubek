//! How the norm's operands are bound.

use cubek_microkernel::rms_norm::GateOrder;
use cubek_rms_norm::kernels::{RmsNormCall, RmsNormGate, RmsNormTensors};
use cubek_std::Hardware;
use cubek_test_utils::{TestOutcome, ValidationResult};

use crate::rms_norm::{EPSILON, F16, Run, view};

/// A gate that is an aligned slice of a wider projection, beside a dense input, is read where it
/// lies: its view keeps a start past its buffer's own.
#[test]
fn an_aligned_gate_slice_is_not_copied() {
    let client = cubecl::test_device().client();
    let run = Run {
        element: F16,
        output_element: F16,
        order: Some(GateOrder::GateThenNorm),
    };
    if !run.runs_on(&client) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the device does not run f16".to_string(),
        ))
        .enforce();
        return;
    }
    let (rows, dim) = (5, 1024);
    let ones = vec![1.0; rows * dim];
    let call = RmsNormCall {
        client: client.clone(),
        input: view(&client, &ones, &[rows, dim], (dim, 0), F16),
        weight: view(&client, &ones[..dim], &[dim], (dim, 0), F16),
        epsilon: EPSILON,
        output_element: F16,
        gate: Some(RmsNormGate {
            tensor: view(&client, &ones, &[rows, dim], (3 * dim, dim), F16),
            order: GateOrder::GateThenNorm,
        }),
    }
    .grouped(128);
    let tensors = RmsNormTensors::new(&Hardware::from_client(&client), &call)
        .expect("the device runs the norm");
    assert_ne!(tensors.problem().start_bytes.gate, 0, "the gate was copied");
}
