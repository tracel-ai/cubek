//! The logsumexp every tile routine of cubek-attention writes beside its output on request, against
//! the host's: what the backward pass reads its probabilities back from. Then a backward pass run on
//! a forward's own output and logsumexp.

use super::reference::{Case, Reference};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_attention::kernels::{
    AttentionRoutine, AttentionScoring, AttentionTensors, CubeShuffleAttentionRoutine,
    CubeStagedAttentionRoutine, HypercubeLastArrivalShuffleAttentionRoutine,
    HypercubeSecondLaunchFragmentAttentionRoutine, HypercubeSecondLaunchShuffleAttentionRoutine,
    PlaneFragmentAttentionRoutine, ScoreScale, UnitAttentionRoutine,
};
use cubek_attention_backward::AttentionBackwardTensors;
use cubek_attention_backward::kernels::{AttentionBackwardRoutine, UnitAttentionBackwardRoutine};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneUnitGroupAttentionMicrokernel,
};
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionVisibility};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// A prompt chunk of two groups of two query heads: no step of queries or keys divides it evenly.
const PROMPT: Case = Case {
    batch: 1,
    heads: 4,
    kv_heads: 2,
    seq_q: 13,
    seq_kv: 37,
    head_dim: 32,
    val_dim: 32,
    visibility: AttentionVisibility::Causal,
};

/// A single position of the same heads, which the forward walks the group of as its queries.
const DECODE: Case = Case {
    seq_q: 1,
    visibility: AttentionVisibility::Attended,
    ..PROMPT
};

/// `reference`'s inputs as a forward attention's tensors, asking for each query's logsumexp.
fn forward_tensors(client: &Client, case: Case, reference: &Reference) -> AttentionTensors {
    let [query, key, value] = reference.inputs.clone().map(PlainTensor::of_handle);
    let lengths = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(u32::elem_type_native())
        .custom(vec![case.seq_kv as f32])
        .generate_without_host_data();
    AttentionTensors::new(
        client,
        query,
        key,
        value,
        PlainTensor::of_handle(lengths),
        AttentionScoring {
            visibility: case.visibility,
            scale: ScoreScale::Custom(case.scale() as f32),
        },
    )
    .expect("the case's tensors are verified")
    .with_logsumexp()
}

/// `routine` at `settings` on `case`, its logsumexp against the host's; a routine this device
/// declines is skipped, and says so.
fn check_logsumexp<R: AttentionRoutine<M, SM>, M: AttentionMicrokernel, SM>(
    name: &str,
    routine: R,
    settings: Option<R::Settings>,
    case: Case,
) {
    let client: Client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let element = half::f16::elem_type_native();
    let reference = Reference::new(&client, case, element);
    let tensors = forward_tensors(&client, case, &reference);
    let logsumexp = tensors.logsumexp().expect("asked for");
    let Some(settings) = settings.or_else(|| routine.settings(&hw).into_iter().next()) else {
        eprintln!("{name}: declined on this device, no settings");
        return;
    };
    match routine.launch_on(&hw, tensors, &settings) {
        Ok(_) => {
            let actual = HostData::from_tensor_handle(&client, logsumexp, HostDataType::F32);
            for (at, &want) in reference.logsumexp.iter().enumerate() {
                let got = actual.data.get_f32(at) as f64;
                assert!(
                    (got - want).abs() <= 1e-2 * (1. + want.abs()),
                    "{name} {settings}: logsumexp[{at}] = {got}, expected {want}"
                );
            }
        }
        Err(Refusal::Declined { why }) => eprintln!("{name}: declined on this device, {why}"),
        Err(refusal) => eprintln!("{name}: declined on this device, {refusal:?}"),
    }
}

#[test]
fn the_unit_routine_writes_the_logsumexp() {
    check_logsumexp("unit", UnitAttentionRoutine, None, PROMPT);
    check_logsumexp("unit", UnitAttentionRoutine, None, DECODE);
}

#[test]
fn the_plane_routine_writes_the_logsumexp() {
    check_logsumexp("plane", PlaneFragmentAttentionRoutine, None, PROMPT);
}

/// Planes that split the same rows' keys merge their state before they write it.
#[test]
fn the_cube_staged_routines_write_the_logsumexp() {
    let fragments = CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new();
    check_logsumexp("cube-fragment", fragments, None, PROMPT);
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let split = fragments
        .settings(&hw)
        .into_iter()
        .find(|settings| settings.microkernel.planes_along_keys.get() > 1);
    if let Some(split) = split {
        check_logsumexp("cube-fragment split keys", fragments, Some(split), PROMPT);
    }
    let unit_groups = CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new();
    check_logsumexp("cube-unit-group", unit_groups, None, PROMPT);
}

/// The lead's planes split one head's keys and merge their rows from slots.
#[test]
fn the_cube_shuffle_routine_writes_the_logsumexp() {
    check_logsumexp("cube-shuffle", CubeShuffleAttentionRoutine, None, PROMPT);
    check_logsumexp("cube-shuffle", CubeShuffleAttentionRoutine, None, DECODE);
}

/// Cubes split a box's keys and a merge writes the rows.
#[test]
fn the_hypercube_routines_write_the_logsumexp() {
    check_logsumexp(
        "hypercube-second-launch-shuffle",
        HypercubeSecondLaunchShuffleAttentionRoutine::new(),
        None,
        DECODE,
    );
    check_logsumexp(
        "hypercube-second-launch-fragment",
        HypercubeSecondLaunchFragmentAttentionRoutine,
        None,
        PROMPT,
    );
    check_logsumexp(
        "hypercube-last-arrival-shuffle",
        HypercubeLastArrivalShuffleAttentionRoutine::new(),
        None,
        DECODE,
    );
}

/// A backward pass reading the forward's own output, `[b, q, h, dv]` viewed as `[b, h, q, dv]`,
/// and its logsumexp: the gradients are the host's.
#[test]
fn the_backward_pass_reads_the_forward_it_follows() {
    let client: Client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = PROMPT;
    let element = half::f16::elem_type_native();
    let reference = Reference::new(&client, case, element);
    let tensors = forward_tensors(&client, case, &reference);
    let logsumexp = tensors.logsumexp().expect("asked for");
    let out = UnitAttentionRoutine
        .launch_lead_on(&hw, tensors)
        .expect("the unit routine runs everywhere");
    let [batch, queries, heads, value] = out.shape().dims::<4>();
    let strides = out.strides().clone();
    let out = PlainTensor::of_handle(out).viewed(Metadata::new(
        [batch, heads, queries, value],
        [strides[0], strides[2], strides[1], strides[3]],
    ));
    let [query, key, value] = reference.inputs.clone().map(PlainTensor::of_handle);
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
    let routine = UnitAttentionBackwardRoutine;
    let settings = routine.lead(&hw);
    let gradients = routine
        .launch_on(&hw, backward, &settings)
        .expect("the unit routine runs everywhere");
    reference.check(&client, gradients, 3e-2);
}
