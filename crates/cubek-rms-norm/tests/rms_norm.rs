//! Every RMS-norm routine, at every setting it lists, against a host reference computed in `f64`,
//! at shapes chosen to be awkward: a decode step's model row, a prompt's head rows, a short row, a
//! prime row that binds one element at a time, a row one line past a power of two, a row longer
//! than a plane's registers hold, no rows at all, and a weight with a row for each row.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_rms_norm::kernels::{
    CubeGroupsRmsNormRoutine, CubeRmsNormRoutine, PlaneRmsNormRoutine, RmsNormCall, RmsNormRoutine,
    RmsNormRoutineCandidate,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, TestInput};

const EPSILON: f32 = 1e-6;

/// The floor serves every case at every setting: a decline would be a family with nothing to fall
/// back on.
#[test]
fn cube_groups_routine_matches_the_reference() {
    check_routine(CubeGroupsRmsNormRoutine, Declines::Never);
}

/// A plane serves every row its lanes hold, and declines the rest by its microkernel's reasons.
#[test]
fn plane_routine_matches_the_reference() {
    check_routine(PlaneRmsNormRoutine, Declines::AsItsMicrokernelStates);
}

/// A cube serves every row its planes' runs fit, and declines the rest by its microkernel's
/// reasons.
#[test]
fn cube_routine_matches_the_reference() {
    check_routine(CubeRmsNormRoutine, Declines::AsItsMicrokernelStates);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Declines {
    Never,
    AsItsMicrokernelStates,
}

/// `rows` rows of `dim`, scaled by one weight row, or by a weight row for each row.
#[derive(Debug, Clone, Copy)]
struct Case {
    rows: usize,
    dim: usize,
    weight_a_row: bool,
}

impl Case {
    const fn new(rows: usize, dim: usize) -> Self {
        Self {
            rows,
            dim,
            weight_a_row: false,
        }
    }

    const fn weight_a_row(self) -> Self {
        Self {
            weight_a_row: true,
            ..self
        }
    }

    fn input(&self) -> Vec<f32> {
        (0..self.rows * self.dim).map(|i| wobble(i, 1)).collect()
    }

    fn weight(&self) -> Vec<f32> {
        let rows = if self.weight_a_row { self.rows } else { 1 };
        (0..rows * self.dim).map(|i| wobble(i, 2) / 2.0).collect()
    }

    /// The norm of `input`'s rows scaled by `weight`, both as `element` holds them, in `f64`.
    fn reference(&self, element: ElemType) -> Vec<f64> {
        let (input, weight) = (self.input(), self.weight());
        let mut out = Vec::with_capacity(input.len());
        for row in 0..self.rows {
            let values: Vec<f64> = input[row * self.dim..(row + 1) * self.dim]
                .iter()
                .map(|&x| round_to(x, element) as f64)
                .collect();
            let mean = values.iter().map(|x| x * x).sum::<f64>() / self.dim as f64;
            let scale = 1.0 / (mean + EPSILON as f64).sqrt();
            let weight_row = if self.weight_a_row { row } else { 0 };
            for (d, x) in values.iter().enumerate() {
                let w = round_to(weight[weight_row * self.dim + d], element) as f64;
                out.push(x * scale * w);
            }
        }
        out
    }
}

fn cases() -> Vec<Case> {
    vec![
        // Decode steps: the model's row, and a head norm's rows.
        Case::new(1, 2560),
        Case::new(16, 128),
        // A prompt's rows of the model and of its heads.
        Case::new(37, 1024),
        Case::new(300, 128),
        // A short row, a prime one that binds one element at a time, and one a line past a power
        // of two.
        Case::new(3, 96),
        Case::new(2, 111),
        Case::new(1, 4100),
        // A row longer than a plane's registers hold, which only the cube and the floor take.
        Case::new(2, 8192),
        // No rows: nothing to launch.
        Case::new(0, 128),
        // A weight a row, as the query and key heads of a decode step share their norms' weights.
        Case::new(6, 128).weight_a_row(),
    ]
}

/// `routine` at every setting it lists, in `f32` and `f16`, on every case, against the reference.
fn check_routine<M: Microkernel, R: RmsNormRoutine<M> + Copy>(routine: R, declines: Declines) {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let mut launched = 0;
    for candidate in RmsNormRoutineCandidate::list(routine, &hw) {
        let settings = candidate.settings();
        for (element, tolerance) in [
            (ElemType::Float(FloatKind::F32), 1e-5),
            (ElemType::Float(FloatKind::F16), 4e-3),
        ] {
            for case in cases() {
                let call = RmsNormCall {
                    client: client.clone(),
                    input: tensor(&client, case.input(), &[case.rows, case.dim], element),
                    weight: tensor(&client, case.weight(), &[case.weight().len()], element),
                    epsilon: EPSILON,
                };
                let out = match (candidate.launch(&call), declines) {
                    (Ok(out), _) => out,
                    (Err(why), Declines::AsItsMicrokernelStates)
                        if why == (Refusal::Declined { why: M::DECLINE }) =>
                    {
                        continue;
                    }
                    (Err(why), _) => panic!("{settings} {case:?} at {element:?}: {why:?}"),
                };
                launched += 1;
                // No rows is no launch, and an empty buffer has nothing to read back.
                if case.rows == 0 {
                    assert_eq!(out.shape().num_elements(), 0, "{settings} {case:?}");
                    continue;
                }
                let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
                for (index, want) in case.reference(element).into_iter().enumerate() {
                    let got = got.data.get_f32(index) as f64;
                    assert!(
                        (got - want).abs() <= tolerance * (1.0 + want.abs()),
                        "{settings} {case:?} at {element:?}, element {index}: got {got}, want {want}"
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

/// Values in `[-2, 2)`, not periodic in any extent a case uses.
fn wobble(i: usize, salt: usize) -> f32 {
    ((i.wrapping_mul(2654435761) + salt * 40503) % 2048) as f32 / 512.0 - 2.0
}

/// `value` as `element` holds it.
fn round_to(value: f32, element: ElemType) -> f32 {
    match element {
        ElemType::Float(FloatKind::F16) => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}
