//! Runs a routine over the blocks of runs and cases, against the reference.

use super::case::{EPSILON, cases};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_rms_norm::kernels::{RmsNormCall, RmsNormRoutine, RmsNormRoutineCandidate};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, TestInput};

const F32: ElemType = ElemType::Float(FloatKind::F32);
const F16: ElemType = ElemType::Float(FloatKind::F16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Declines {
    Never,
    AsItsMicrokernelStates,
}

/// `routine` at every setting it lists, in `f32` and `f16`, on every case, against the reference.
pub fn check_routine<M: Microkernel, R: RmsNormRoutine<M> + Copy>(routine: R, declines: Declines) {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let mut launched = 0;
    for candidate in RmsNormRoutineCandidate::list(routine, &hw) {
        let settings = candidate.settings();
        for (element, tolerance) in [(F32, 1e-5), (F16, 4e-3)] {
            for case in cases() {
                let weight = case.weight();
                let call = RmsNormCall {
                    client: client.clone(),
                    input: tensor(&client, case.input(), &[case.rows, case.dim], element),
                    weight: tensor(&client, weight.clone(), &[weight.len()], element),
                    epsilon: EPSILON,
                };
                let what = format!("{settings} {case:?} at {element:?}");
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
                let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                for (index, want) in case.reference(element).into_iter().enumerate() {
                    let got = got.data.get_f32(index) as f64;
                    assert!(
                        (got - want).abs() <= tolerance * (1.0 + want.abs()),
                        "{what}, element {index}: got {got}, want {want}"
                    );
                }
            }
        }
    }
    assert!(launched > 0, "the routine ran no case on this device");
}

/// `values` uploaded as `element`, row-major in `shape`.
fn tensor(client: &Client, values: Vec<f32>, shape: &[usize], element: ElemType) -> PlainTensor {
    let handle = TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(element)
        .custom(values)
        .generate_without_host_data();
    PlainTensor::of_handle(handle)
}
