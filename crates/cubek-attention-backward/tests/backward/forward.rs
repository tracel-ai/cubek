//! The output every tile routine of cubek-attention writes, and the logsumexp it writes beside it on
//! request, against the host's: the logsumexp is what the backward pass reads its probabilities
//! back from. Then a backward pass run on a forward's own output and logsumexp.

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
use cubek_microkernel::attention::{AttentionMicrokernel, AttentionVisibility, QueryPlacement};
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
    visibility: AttentionVisibility::Causal {
        queries: QueryPlacement::Last,
    },
    hidden: None,
};

/// A causal prompt chunk at a head dim of 128, the one a model's prefill runs: four query heads to
/// a key head, and queries and keys no box of either divides.
const PROMPT_WIDE: Case = Case {
    heads: 8,
    kv_heads: 2,
    seq_q: 70,
    seq_kv: 300,
    head_dim: 128,
    val_dim: 128,
    ..PROMPT
};

/// A single position of the same heads, which the forward walks the group of as its queries.
const DECODE: Case = Case {
    seq_q: 1,
    visibility: AttentionVisibility::Attended,
    ..PROMPT
};

/// The prompt with its queries placed first among the keys: query `i` sees the keys up to `i`.
const PROMPT_PLACED_FIRST: Case = Case {
    visibility: AttentionVisibility::Causal {
        queries: QueryPlacement::First,
    },
    ..PROMPT
};

/// A mask that differs from head to head, so a routine reading another head's cells is caught. It
/// never hides the first key, which a query placed first may be the only one to see.
fn stripes([_batch, head, query, key]: [usize; 4]) -> bool {
    key > 0 && (query + 2 * key + 3 * head).is_multiple_of(5)
}

/// The prompt under a mask, on top of its causal diagonal.
const PROMPT_MASKED: Case = Case {
    hidden: Some(stripes),
    ..PROMPT
};

/// The prompt under a mask, its queries placed first.
const PROMPT_MASKED_PLACED_FIRST: Case = Case {
    hidden: Some(stripes),
    ..PROMPT_PLACED_FIRST
};

/// A single position under a mask, its group's heads walked as its queries.
const DECODE_MASKED: Case = Case {
    hidden: Some(stripes),
    ..DECODE
};

/// `reference`'s inputs as a forward attention's tensors, asking for each query's logsumexp,
/// under the case's mask where it has one, stored as `u32` cells.
fn forward_tensors(client: &Client, case: Case, reference: &Reference) -> AttentionTensors {
    let [query, key, value] = reference.inputs.clone().map(PlainTensor::of_handle);
    let lengths = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(u32::elem_type_native())
        .custom(vec![case.seq_kv as f32])
        .generate_without_host_data();
    let tensors = AttentionTensors::new(
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
    .with_logsumexp();
    match case.hidden {
        Some(hidden) => {
            let (batch, heads, queries, keys) = (case.batch, case.heads, case.seq_q, case.seq_kv);
            let mut cells = Vec::with_capacity(batch * heads * queries * keys);
            for b in 0..batch {
                for h in 0..heads {
                    for i in 0..queries {
                        for j in 0..keys {
                            cells.push(hidden([b, h, i, j]) as u32 as f32);
                        }
                    }
                }
            }
            let mask =
                TestInput::builder(client.clone(), Shape::new([batch, heads, queries, keys]))
                    .dtype(u32::elem_type_native())
                    .custom(cells)
                    .generate_without_host_data();
            tensors.with_mask(PlainTensor::of_handle(mask))
        }
        None => tensors,
    }
}

/// `routine` at `settings` on `case`, its output and its logsumexp against the host's; a routine
/// this device declines is skipped, and says so.
fn check_forward<R: AttentionRoutine<M, SM>, M: AttentionMicrokernel, SM>(
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
        Ok(out) => {
            // The output comes back `[b, q, h, dv]`, the host's `[b, h, q, dv]`.
            let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
            for (at, &want) in reference.out.iter().enumerate() {
                let c = at % case.val_dim;
                let i = at / case.val_dim % case.seq_q;
                let h = at / (case.val_dim * case.seq_q) % case.heads;
                let b = at / (case.val_dim * case.seq_q * case.heads);
                let got = actual.get_f32(&[b, i, h, c]) as f64;
                assert!(
                    (got - want).abs() <= 5e-3 * (1. + want.abs()),
                    "{name} {settings}: out[{b}, {h}, {i}, {c}] = {got}, expected {want}"
                );
            }
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
    check_forward("unit", UnitAttentionRoutine, None, PROMPT);
    check_forward("unit", UnitAttentionRoutine, None, DECODE);
}

#[test]
fn the_plane_routine_writes_the_logsumexp() {
    check_forward("plane", PlaneFragmentAttentionRoutine, None, PROMPT);
}

/// Planes that split the same rows' keys merge their state before they write it.
#[test]
fn the_cube_staged_routine_writes_the_logsumexp() {
    let fragments = CubeStagedAttentionRoutine;
    check_forward("cube-fragment", fragments, None, PROMPT);
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let split = fragments
        .settings(&hw)
        .into_iter()
        .find(|settings| settings.microkernel.planes_along_keys.get() > 1);
    if let Some(split) = split {
        check_forward("cube-fragment split keys", fragments, Some(split), PROMPT);
    }
}

/// The lead's planes split one head's keys and merge their rows from slots.
/// Every setting the cube staged routine lists, the copy engine's among them, at a model's head
/// dim: the query's and the stages' layouts, the rings and the run of score fragments the value
/// reads are each a different kernel.
#[test]
fn every_cube_staged_setting_matches_the_host_at_a_head_dim_of_128() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    for settings in CubeStagedAttentionRoutine.settings(&hw) {
        check_forward(
            "cube-fragment",
            CubeStagedAttentionRoutine,
            Some(settings),
            PROMPT_WIDE,
        );
    }
}

/// Every setting the hypercube of plane fragments lists at a model's head dim, each box's keys cut
/// across its cubes: the stages each cube hands on to the partial it leaves are each a different
/// kernel.
#[test]
fn every_hypercube_fragment_setting_matches_the_host_at_a_head_dim_of_128() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    for settings in HypercubeSecondLaunchFragmentAttentionRoutine.settings(&hw) {
        check_forward(
            "hypercube-second-launch-fragment",
            HypercubeSecondLaunchFragmentAttentionRoutine,
            Some(settings),
            PROMPT_WIDE,
        );
    }
}

#[test]
fn the_cube_shuffle_routine_writes_the_logsumexp() {
    check_forward("cube-shuffle", CubeShuffleAttentionRoutine, None, PROMPT);
    check_forward("cube-shuffle", CubeShuffleAttentionRoutine, None, DECODE);
}

/// Cubes split a box's keys and a merge writes the rows.
#[test]
fn the_hypercube_routines_write_the_logsumexp() {
    check_forward(
        "hypercube-second-launch-shuffle",
        HypercubeSecondLaunchShuffleAttentionRoutine::new(),
        None,
        DECODE,
    );
    check_forward(
        "hypercube-second-launch-fragment",
        HypercubeSecondLaunchFragmentAttentionRoutine,
        None,
        PROMPT,
    );
    check_forward(
        "hypercube-last-arrival-shuffle",
        HypercubeLastArrivalShuffleAttentionRoutine::new(),
        None,
        DECODE,
    );
}

/// Every routine of many positions hides what a diagonal placed first and a mask hide, alone and
/// together: its logsumexp is the host's over the keys the case leaves seen.
#[test]
fn every_prompt_routine_hides_what_a_mask_and_a_first_placed_diagonal_hide() {
    for case in [
        PROMPT_PLACED_FIRST,
        PROMPT_MASKED,
        PROMPT_MASKED_PLACED_FIRST,
    ] {
        check_forward("unit", UnitAttentionRoutine, None, case);
        check_forward("plane", PlaneFragmentAttentionRoutine, None, case);
        check_forward("cube-fragment", CubeStagedAttentionRoutine, None, case);
        check_forward("cube-shuffle", CubeShuffleAttentionRoutine, None, case);
        check_forward(
            "hypercube-second-launch-fragment",
            HypercubeSecondLaunchFragmentAttentionRoutine,
            None,
            case,
        );
    }
}

/// Every routine of a single position hides what a mask hides, each head of the group reading
/// its own cells.
#[test]
fn every_decode_routine_hides_what_a_mask_hides() {
    check_forward("unit", UnitAttentionRoutine, None, DECODE_MASKED);
    check_forward(
        "cube-shuffle",
        CubeShuffleAttentionRoutine,
        None,
        DECODE_MASKED,
    );
    check_forward(
        "hypercube-second-launch-shuffle",
        HypercubeSecondLaunchShuffleAttentionRoutine::new(),
        None,
        DECODE_MASKED,
    );
    check_forward(
        "hypercube-last-arrival-shuffle",
        HypercubeLastArrivalShuffleAttentionRoutine::new(),
        None,
        DECODE_MASKED,
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
