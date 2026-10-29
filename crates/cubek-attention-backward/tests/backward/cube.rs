//! The cube staged routine against the host's backward pass, where the device offers it a matrix
//! instruction.

use super::reference::{Case, Reference};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubek_attention_backward::kernels::{
    AttentionBackwardRoutine, CubeStagedAttentionBackwardRoutine,
    CubeStagedAttentionBackwardRoutineSettings,
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
