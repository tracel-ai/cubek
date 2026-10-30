//! The hypercube routine against the host's backward pass, where the device offers it a matrix
//! instruction and `f32` atomic adds.

use super::reference::{Case, Reference};
use cubecl::prelude::*;
use cubek_attention_backward::kernels::{
    AttentionBackwardRoutine, HypercubeAttentionBackwardRoutine,
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
fn the_hypercube_routine_matches_the_reference() {
    Reference::run(
        HypercubeAttentionBackwardRoutine,
        CASE,
        f32::elem_type_native(),
        1e-3,
        None,
    );
}

/// Four query heads over two key heads: a key's gradient sums every query head of its group, and
/// a query's gradient every key cube of its key head.
#[test]
fn the_hypercube_routine_sums_a_group_into_its_key_head() {
    Reference::run(
        HypercubeAttentionBackwardRoutine,
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

/// No query sees past its own position: the keys' walk starts at the first query that sees them,
/// so a query's gradient sums only the key cubes it sees.
#[test]
fn the_hypercube_routine_masks_causally() {
    Reference::run(
        HypercubeAttentionBackwardRoutine,
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
fn the_hypercube_routine_in_f16() {
    Reference::run(
        HypercubeAttentionBackwardRoutine,
        CASE,
        half::f16::elem_type_native(),
        3e-2,
        None,
    );
}

/// Every setting the routine lists prepares and computes the same gradients, on groups of two
/// heads and a value dim two fragments wide.
#[test]
fn every_hypercube_setting_matches_the_reference() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        heads: 4,
        val_dim: 16,
        ..CASE
    };
    for settings in HypercubeAttentionBackwardRoutine.settings(&hw) {
        Reference::run(
            HypercubeAttentionBackwardRoutine,
            case,
            f32::elem_type_native(),
            1e-3,
            Some(settings),
        );
    }
}

/// Every setting in `f16` on the same case: what the settings holding the score in their
/// fragments run on, the manual `mma` taking 16-bit operands only, each step leaving the score's
/// gradient in a window for the share.
#[test]
fn every_hypercube_setting_matches_the_reference_in_f16() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        heads: 4,
        val_dim: 16,
        ..CASE
    };
    for settings in HypercubeAttentionBackwardRoutine.settings(&hw) {
        Reference::run(
            HypercubeAttentionBackwardRoutine,
            case,
            half::f16::elem_type_native(),
            3e-2,
            Some(settings),
        );
    }
}
