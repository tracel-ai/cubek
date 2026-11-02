//! The cube staged routine against the host's backward pass, where the device offers it a matrix
//! instruction.

use super::reference::{Case, Reference};
use cubecl::prelude::*;
use cubek_attention_backward::kernels::{
    AttentionBackwardRoutine, CubeStagedAttentionBackwardRoutine,
};
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
use cubek_std::Hardware;

/// Two heads of their own key heads, more keys than queries, and no cube's box that divides the
/// queries or the keys evenly: every walk's last step, and a cube's last plane, is cut short.
const CASE: Case = Case {
    batch: 1,
    heads: 2,
    kv_heads: 2,
    seq_q: 13,
    seq_kv: 21,
    head_dim: 16,
    val_dim: 8,
    visibility: AttentionVisibility::Attended,
    hidden: None,
};

#[test]
fn the_cube_routine_matches_the_reference() {
    Reference::run(
        CubeStagedAttentionBackwardRoutine,
        CASE,
        f32::elem_type_native(),
        1e-3,
        None,
    );
}

/// Four query heads over two key heads: a key's gradient sums every query head of its group.
#[test]
fn the_cube_routine_sums_a_group_into_its_key_head() {
    Reference::run(
        CubeStagedAttentionBackwardRoutine,
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
fn the_cube_routine_masks_causally() {
    Reference::run(
        CubeStagedAttentionBackwardRoutine,
        Case {
            seq_q: 21,
            visibility: AttentionVisibility::Causal {
                queries: QueryPlacement::Last,
            },
            ..CASE
        },
        f32::elem_type_native(),
        1e-3,
        None,
    );
}

#[test]
fn the_cube_routine_in_f16() {
    Reference::run(
        CubeStagedAttentionBackwardRoutine,
        CASE,
        half::f16::elem_type_native(),
        3e-2,
        None,
    );
}

/// Every setting the routine lists prepares and computes the same gradients, on groups of two
/// heads, so a cube holding two heads holds a whole group, and a value dim two fragments wide.
#[test]
fn every_cube_setting_matches_the_reference() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        heads: 4,
        val_dim: 16,
        ..CASE
    };
    for settings in CubeStagedAttentionBackwardRoutine.settings(&hw) {
        Reference::run(
            CubeStagedAttentionBackwardRoutine,
            case,
            f32::elem_type_native(),
            1e-3,
            Some(settings),
        );
    }
}

/// Every setting in `f16` on the same case: what the settings holding the score in their
/// fragments run on, the manual `mma` taking 16-bit operands only.
#[test]
fn every_cube_setting_matches_the_reference_in_f16() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        heads: 4,
        val_dim: 16,
        ..CASE
    };
    for settings in CubeStagedAttentionBackwardRoutine.settings(&hw) {
        Reference::run(
            CubeStagedAttentionBackwardRoutine,
            case,
            half::f16::elem_type_native(),
            3e-2,
            Some(settings),
        );
    }
}

/// Every setting holding both walks' scores in fragments, under a causal mask, on queries and keys
/// spanning several stages and several boxes: the key walks start past their first stages and the
/// query walks stop short of their last, which a case shorter than one stage never shows, and the
/// heads of a group set apart add their keys' gradients into one sum.
#[test]
fn every_held_cube_setting_masks_causally_over_several_stages_in_f16() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        heads: 4,
        seq_q: 150,
        seq_kv: 150,
        val_dim: 16,
        visibility: AttentionVisibility::Causal {
            queries: QueryPlacement::Last,
        },
        ..CASE
    };
    let held = CubeStagedAttentionBackwardRoutine
        .settings(&hw)
        .into_iter()
        .filter(|settings| {
            settings.query_gradient.holds_its_score() && settings.key_gradient.holds_its_score()
        });
    for settings in held {
        Reference::run(
            CubeStagedAttentionBackwardRoutine,
            case,
            half::f16::elem_type_native(),
            3e-2,
            Some(settings),
        );
    }
}
