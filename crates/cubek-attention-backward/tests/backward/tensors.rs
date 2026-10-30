//! What the backward pass's tensors refuse to be.

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubek_attention_backward::AttentionBackwardTensors;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::AttentionVisibility;
use cubek_test_utils::TestInput;

/// A logsumexp of the wrong rank stops at the message naming the shapes, before any of its dims
/// are read.
#[test]
#[should_panic(expected = "the output and its gradient are [b, h, q, dv], the logsumexp [b, h, q]")]
fn a_logsumexp_of_the_wrong_rank_is_named() {
    let client: Client = cubecl::test_device().client();
    let tensor = |shape: &[usize], element: ElemType| {
        PlainTensor::of_handle(
            TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                .dtype(element)
                .arange()
                .generate_without_host_data(),
        )
    };
    let element = half::f16::elem_type_native();
    let sum = f32::elem_type_native();
    let _ = AttentionBackwardTensors::new(
        &client,
        tensor(&[1, 2, 4, 8], element),
        tensor(&[1, 2, 4, 8], element),
        tensor(&[1, 2, 4, 8], element),
        tensor(&[1, 2, 4, 8], element),
        tensor(&[1, 2, 4, 8], element),
        // `[b, h, q, 1]`, one dim too many.
        tensor(&[1, 2, 4, 1], sum),
        1.0,
        AttentionVisibility::Attended,
    );
}
