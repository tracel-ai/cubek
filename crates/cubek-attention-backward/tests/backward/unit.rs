//! The unit routine against the host's backward pass.

use super::reference::{Case, Reference};
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubek_attention_backward::AttentionBackwardRoutine;
use cubek_attention_backward::kernels::UnitAttentionBackwardRoutine;
use cubek_microkernel::attention::AttentionVisibility;
use cubek_std::Hardware;

/// Two heads of their own key heads, more keys than queries, and no step that divides the
/// queries or the keys evenly: every walk's last step is cut short.
const CASE: Case = Case {
    batch: 1,
    heads: 2,
    kv_heads: 2,
    seq_q: 13,
    seq_kv: 21,
    head_dim: 16,
    val_dim: 8,
    visibility: AttentionVisibility::Attended,
};

/// The routine at its lead setting on `case`, in `element`, against the host within `tolerance`.
fn run(case: Case, element: ElemType, tolerance: f64) {
    let client: Client = cubecl::test_device().client();
    let reference = Reference::new(&client, case, element);
    let hw = Hardware::from_client(&client);
    let routine = UnitAttentionBackwardRoutine;
    let settings = routine.settings(&hw).swap_remove(0);
    let gradients = routine
        .launch_on(&hw, reference.tensors.clone(), &settings)
        .expect("the unit routine runs everywhere");
    reference.check(&client, gradients, tolerance);
}

#[test]
fn the_unit_routine_matches_the_reference() {
    run(CASE, f32::elem_type_native(), 1e-4);
}

/// Four query heads over two key heads: a key's gradient sums every query head of its group.
#[test]
fn the_unit_routine_sums_a_group_into_its_key_head() {
    run(
        Case {
            heads: 4,
            batch: 2,
            ..CASE
        },
        f32::elem_type_native(),
        1e-4,
    );
}

/// No query sees past its own position: the keys' walk starts at the first query that sees them.
#[test]
fn the_unit_routine_masks_causally() {
    run(
        Case {
            seq_q: 21,
            visibility: AttentionVisibility::Causal,
            ..CASE
        },
        f32::elem_type_native(),
        1e-4,
    );
}

/// A sliding window bounds the keys a query sees from below too: the bias masks them.
#[test]
fn the_unit_routine_masks_a_sliding_window() {
    run(
        Case {
            seq_q: 21,
            visibility: AttentionVisibility::Windowed {
                window: NonZeroUsize::new(5).unwrap(),
            },
            ..CASE
        },
        f32::elem_type_native(),
        1e-4,
    );
}

#[test]
fn the_unit_routine_in_f16() {
    run(CASE, half::f16::elem_type_native(), 3e-2);
}

/// Every setting the routine lists prepares and computes the same gradients.
#[test]
fn every_unit_setting_matches_the_reference() {
    let client: Client = cubecl::test_device().client();
    let reference = Reference::new(&client, CASE, f32::elem_type_native());
    let hw = Hardware::from_client(&client);
    let routine = UnitAttentionBackwardRoutine;
    for settings in routine.settings(&hw) {
        let gradients = routine
            .launch_on(&hw, reference.tensors.clone(), &settings)
            .unwrap_or_else(|refusal| panic!("{settings}: {refusal:?}"));
        reference.check(&client, gradients, 1e-4);
    }
}
