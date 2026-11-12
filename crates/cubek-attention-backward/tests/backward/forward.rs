//! A backward pass run on a forward's own output and logsumexp: the logsumexp is what the backward
//! pass reads its probabilities back from. The forward routines are cubek-attention's to test.

use super::reference::{Case, Reference};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_attention::kernels::{
    AttentionLaunch, AttentionScoring, AttentionTensors, DirectAttention, ScoreScale,
    UnitAttentionRoutine,
};
use cubek_attention_backward::AttentionBackwardTensors;
use cubek_attention_backward::kernels::{AttentionBackwardStrategy, UnitAttentionBackward};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
use cubek_std::Hardware;
use cubek_test_utils::TestInput;

/// A prompt chunk of two groups of two query heads: no step of queries or keys divides it evenly.
const PROMPT: Case = Case {
    batch: 1,
    heads: 4,
    kv_heads: 2,
    seq_q: 13,
    seq_kv: 37,
    head_dim: 32,
    val_dim: 32,
    visibility: AttentionVisibility::Causal {
        queries: QueryPlacement::Last,
    },
    hidden: None,
};

/// A backward pass reading the forward's own output, `[b, q, h, dv]` viewed as `[b, h, q, dv]`,
/// and its logsumexp: the gradients are the host's.
#[test]
fn the_backward_pass_reads_the_forward_it_follows() {
    let client: Client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = PROMPT;
    let element = half::f16::elem_type_native();
    let reference = Reference::new(&client, case, element);
    let [query, key, value] = reference.inputs.clone().map(PlainTensor::of_handle);
    let lengths = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(u32::elem_type_native())
        .custom(vec![case.seq_kv as f32])
        .generate_without_host_data();
    let tensors = AttentionTensors::new(
        &client,
        query.clone(),
        key.clone(),
        value.clone(),
        PlainTensor::of_handle(lengths),
        AttentionScoring {
            visibility: case.visibility,
            scale: ScoreScale::Custom(case.scale() as f32),
        },
    )
    .expect("the case's tensors are verified")
    .with_logsumexp();
    let logsumexp = tensors.logsumexp().expect("asked for");
    let out = DirectAttention(UnitAttentionRoutine)
        .launch_lead_on(&hw, tensors)
        .expect("the unit routine runs everywhere");
    let [batch, queries, heads, value_dim] = out.shape().dims::<4>();
    let strides = out.strides().clone();
    let out = PlainTensor::of_handle(out).viewed(Metadata::new(
        [batch, heads, queries, value_dim],
        [strides[0], strides[2], strides[1], strides[3]],
    ));
    let backward = AttentionBackwardTensors::new(
        &client,
        query,
        key,
        value,
        out,
        PlainTensor::of_handle(reference.out_gradient.clone()),
        PlainTensor::of_handle(logsumexp),
        case.scale() as f32,
        case.visibility,
    )
    .expect("the forward's tensors are verified");
    let strategy = UnitAttentionBackward;
    let settings = strategy.lead(&hw);
    let gradients = strategy
        .launch_on(&hw, backward, &settings)
        .expect("the unit strategy runs everywhere");
    reference.check(&client, gradients, 3e-2);
}
