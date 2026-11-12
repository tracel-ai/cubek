//! The plane strategy against the host's backward pass, where the device offers it a matrix
//! instruction.

use super::reference::{Case, Reference};
use cubecl::prelude::*;
use cubek_attention_backward::kernels::{
    AttentionBackwardStrategy, PlaneFragmentAttentionBackward,
};
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
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
    hidden: None,
};

#[test]
fn the_plane_strategy_matches_the_reference() {
    Reference::run(
        PlaneFragmentAttentionBackward,
        CASE,
        f32::elem_type_native(),
        1e-3,
        None,
    );
}

/// Four query heads over two key heads: a key's gradient sums every query head of its group.
#[test]
fn the_plane_strategy_sums_a_group_into_its_key_head() {
    Reference::run(
        PlaneFragmentAttentionBackward,
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
fn the_plane_strategy_masks_causally() {
    Reference::run(
        PlaneFragmentAttentionBackward,
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
fn the_plane_strategy_in_f16() {
    Reference::run(
        PlaneFragmentAttentionBackward,
        CASE,
        half::f16::elem_type_native(),
        3e-2,
        None,
    );
}

/// Every setting the strategy lists prepares and computes the same gradients, on a value dim two
/// fragments wide, so every grid holds more than one fragment along it.
#[test]
fn every_plane_setting_matches_the_reference() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        val_dim: 16,
        ..CASE
    };
    for settings in PlaneFragmentAttentionBackward.settings(&hw) {
        Reference::run(
            PlaneFragmentAttentionBackward,
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
fn every_plane_setting_matches_the_reference_in_f16() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let case = Case {
        val_dim: 16,
        ..CASE
    };
    for settings in PlaneFragmentAttentionBackward.settings(&hw) {
        Reference::run(
            PlaneFragmentAttentionBackward,
            case,
            half::f16::elem_type_native(),
            3e-2,
            Some(settings),
        );
    }
}

/// A value dim of two fragments: the values' gradient holds a grid of its own two fragments wide.
#[test]
fn the_plane_strategy_holds_a_wide_value_dim() {
    Reference::run(
        PlaneFragmentAttentionBackward,
        Case {
            val_dim: 16,
            ..CASE
        },
        f32::elem_type_native(),
        1e-3,
        None,
    );
}
