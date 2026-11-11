use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_test_utils::TestInput;

pub const EPSILON: f32 = 1e-6;

/// `values` uploaded as `element`, row-major in `shape`.
pub fn tensor(
    client: &Client,
    values: Vec<f32>,
    shape: &[usize],
    element: ElemType,
) -> PlainTensor {
    let handle = TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(element)
        .custom(values)
        .generate_without_host_data();
    PlainTensor::of_handle(handle)
}

/// Values in `[-2, 2)`, not periodic in any extent a case uses.
pub fn wobble(i: usize, salt: usize) -> f32 {
    ((i.wrapping_mul(2654435761) + salt * 40503) % 2048) as f32 / 512.0 - 2.0
}

/// `value` as `element` holds it.
pub fn round_to(value: f32, element: ElemType) -> f32 {
    match element {
        ElemType::Float(FloatKind::F16) => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}
