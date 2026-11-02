//! The SwiGLU cube routine, at every setting it lists, and the autotune, against a host reference
//! computed in `f64`, at shapes chosen to be awkward: decode steps of real models' `d_ff`, a
//! prompt's rows, a short half, a prime one that binds one element at a time, one a line past a
//! power of two, a prime half longer than a group unrolls, and no rows or no width at all.

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;
use cubek_swiglu::kernels::{CubeSwigluRoutine, SwigluCall, SwigluRoutineCandidate, launch_tuned};
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// Every setting serves every case: the routine is its own autotune's floor.
#[test]
fn cube_routine_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for candidate in SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw) {
        let settings = candidate.settings();
        for element in ELEMENTS {
            for case in cases() {
                let out = candidate
                    .launch(&case.call(&client, element))
                    .unwrap_or_else(|why| panic!("{settings} {case:?} at {element:?}: {why:?}"));
                case.check(&client, out, element, &settings.to_string());
            }
        }
    }
}

/// Every case through the autotune, which runs whichever setting its round elected.
#[test]
fn the_autotune_matches_the_reference() {
    let device = cubecl::test_device();
    let client = device.client();
    for element in ELEMENTS {
        for case in cases() {
            let out = launch_tuned(case.call(&client, element), &device, GroupPriority::DEFAULT);
            case.check(&client, out, element, "the autotune");
        }
    }
}

const ELEMENTS: [ElemType; 2] = [
    ElemType::Float(FloatKind::F32),
    ElemType::Float(FloatKind::F16),
];

/// `rows` fused rows of `2 · dim`, laid out `[gate | up]`.
#[derive(Debug, Clone, Copy)]
struct Case {
    rows: usize,
    dim: usize,
}

impl Case {
    const fn new(rows: usize, dim: usize) -> Self {
        Self { rows, dim }
    }

    fn call(&self, client: &Client, element: ElemType) -> SwigluCall {
        let values = (0..self.rows * 2 * self.dim).map(wobble).collect();
        let handle = TestInput::builder(client.clone(), Shape::from(vec![self.rows, 2 * self.dim]))
            .dtype(element)
            .custom(values)
            .generate_without_host_data();
        SwigluCall {
            client: client.clone(),
            both: PlainTensor::of_handle(handle),
        }
    }

    /// `silu(gate) · up` of each row's halves, both as `element` holds them, in `f64`.
    fn reference(&self, element: ElemType) -> Vec<f64> {
        let at = |i: usize| round_to(wobble(i), element) as f64;
        let mut out = Vec::with_capacity(self.rows * self.dim);
        for row in 0..self.rows {
            for d in 0..self.dim {
                let gate = at(row * 2 * self.dim + d);
                let up = at(row * 2 * self.dim + self.dim + d);
                out.push(gate / (1.0 + (-gate).exp()) * up);
            }
        }
        out
    }

    /// `out` against the reference: the result rounded once to the element.
    fn check(&self, client: &Client, out: TensorHandle, element: ElemType, what: &str) {
        let tolerance = match element {
            ElemType::Float(FloatKind::F16) => 2e-3,
            _ => 1e-5,
        };
        assert_eq!(
            out.shape().as_slice(),
            [self.rows, self.dim],
            "{what} {self:?}"
        );
        // No elements is no launch, and an empty buffer has nothing to read back.
        if self.rows * self.dim == 0 {
            return;
        }
        let got = HostData::from_tensor_handle(client, out, HostDataType::F32);
        for (index, want) in self.reference(element).into_iter().enumerate() {
            let got = got.data.get_f32(index) as f64;
            assert!(
                (got - want).abs() <= tolerance * (1.0 + want.abs()),
                "{what} {self:?} at {element:?}, element {index}: got {got}, want {want}"
            );
        }
    }
}

fn cases() -> Vec<Case> {
    vec![
        // Decode steps: Llama 3.2 1B's `d_ff`, Qwen3-4B's, and a batch of four of a smaller one.
        Case::new(1, 8192),
        Case::new(1, 9728),
        Case::new(4, 2816),
        // A prompt's rows.
        Case::new(37, 1024),
        // A short half, a prime one that binds one element at a time, one a line past a power of
        // two, and one whose lines are a prime count.
        Case::new(3, 96),
        Case::new(2, 111),
        Case::new(2, 1028),
        Case::new(2, 2 * 1031),
        // A prime half longer than a group unrolls: one run, walked in a loop.
        Case::new(2, 5003),
        // No rows, and rows of no width: nothing to launch.
        Case::new(0, 128),
        Case::new(3, 0),
    ]
}

/// Values in `[-4, 4)`, not periodic in any extent a case uses.
fn wobble(i: usize) -> f32 {
    ((i.wrapping_mul(2654435761) + 40503) % 2048) as f32 / 256.0 - 4.0
}

/// `value` as `element` holds it.
fn round_to(value: f32, element: ElemType) -> f32 {
    match element {
        ElemType::Float(FloatKind::F16) => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}
