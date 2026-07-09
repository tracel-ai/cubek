//! Backend probes for the vectorized addressing contract: a regrouped
//! `&[Vector<E, W>]` slice is **line-unit indexed** (cubecl's `with_vector_size` divides
//! the slice offset/length by the width), so `slice[i]` touches scalars
//! `[i·W, i·W + W)`. The tile engine's layouts must therefore emit line-unit offsets.

use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, assert_equals_approx};

/// Write a broadcast `Vector<f32, 2>` at line index 1 of a regrouped slice; line-unit
/// indexing lands it on scalars 2..4.
#[cube(launch)]
fn write_line_index_one(t: &Tensor<f32>) {
    if UNIT_POS == 0 {
        let mut buf = unsafe { t.as_slice().as_boxed_unchecked() };
        let lines = buf.as_vectorized_mut().with_vector_size_mut::<Const<2>>();
        lines[1] = Vector::<f32, Const<2>>::cast_from(1.0f32);
    }
}

/// The engine-shaped regroup (`Box<[T]> → as_vectorized → with_vector_size`) must index
/// in line units on every backend.
#[test]
fn regrouped_slice_indexes_in_line_units() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (input, _) = TestInput::builder(client.clone(), shape![8])
        .custom(vec![0.0f32; 8])
        .generate_with_f32_host_data();

    write_line_index_one::launch::<TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_single(),
        input.clone().binding().into_tensor_arg(),
    );

    let got = HostData::from_tensor_handle(&client, input, HostDataType::F32);
    let expected = vec![0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let (_, expected) = TestInput::builder(client, shape![8])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce()
}
