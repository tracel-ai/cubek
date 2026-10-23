//! The plane routine against the host's backward pass, where the device offers it a matrix
//! instruction.

use super::reference::{Case, Reference};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubek_attention_backward::AttentionBackwardRoutine;
use cubek_attention_backward::kernels::{
    PlaneFragmentAttentionBackwardRoutine, PlaneFragmentAttentionBackwardRoutineSettings,
};
use cubek_microkernel::attention::AttentionVisibility;
use cubek_std::Hardware;

/// Two heads of their own key heads, more keys than queries, and no plane's box that divides the
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

/// The routine at `settings` on `case`, in `element`, against the host within `tolerance`; skipped,
/// and said so, where the device declines it.
fn run(
    case: Case,
    element: ElemType,
    tolerance: f64,
    settings: Option<PlaneFragmentAttentionBackwardRoutineSettings>,
) {
    let client: Client = cubecl::test_device().client();
    let reference = Reference::new(&client, case, element);
    let hw = Hardware::from_client(&client);
    let routine = PlaneFragmentAttentionBackwardRoutine;
    let settings = settings.unwrap_or_else(|| routine.settings(&hw).swap_remove(0));
    match routine.launch_on(&hw, reference.tensors.clone(), &settings) {
        Ok(gradients) => reference.check(&client, gradients, tolerance),
        Err(refusal) => eprintln!("{settings}: declined on this device, {refusal:?}"),
    }
}

#[test]
fn the_plane_routine_matches_the_reference() {
    run(CASE, f32::elem_type_native(), 1e-3, None);
}

/// Four query heads over two key heads: a key's gradient sums every query head of its group.
#[test]
fn the_plane_routine_sums_a_group_into_its_key_head() {
    run(
        Case {
            heads: 4,
            batch: 2,
            ..CASE
        },
        f32::elem_type_native(),
        1e-3,
        None,
    );
}

/// No query sees past its own position: the keys' walk starts at the first query that sees them.
#[test]
fn the_plane_routine_masks_causally() {
    run(
        Case {
            seq_q: 21,
            visibility: AttentionVisibility::Causal,
            ..CASE
        },
        f32::elem_type_native(),
        1e-3,
        None,
    );
}

#[test]
fn the_plane_routine_in_f16() {
    run(CASE, half::f16::elem_type_native(), 3e-2, None);
}

/// Every setting the routine lists prepares and computes the same gradients, on a value dim two
/// fragments wide, so every grid holds more than one fragment along it.
#[test]
fn every_plane_setting_matches_the_reference() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        val_dim: 16,
        ..CASE
    };
    for settings in PlaneFragmentAttentionBackwardRoutine.settings(&hw) {
        run(case, f32::elem_type_native(), 1e-3, Some(settings));
    }
}

/// A value dim of two fragments: the values' gradient holds a grid of its own two fragments wide.
#[test]
fn the_plane_routine_holds_a_wide_value_dim() {
    run(
        Case {
            val_dim: 16,
            ..CASE
        },
        f32::elem_type_native(),
        1e-3,
        None,
    );
}
