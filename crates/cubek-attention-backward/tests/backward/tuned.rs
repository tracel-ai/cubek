//! The entry point, which races the routines, against the host's backward pass.

use super::reference::{Case, Reference};
use cubecl::prelude::*;
use cubek_attention_backward::attention_backward;
use cubek_microkernel::attention::AttentionVisibility;

/// Four query heads over two key heads, causal, and no box of any routine that divides the
/// queries or the keys evenly.
const CASE: Case = Case {
    batch: 1,
    heads: 4,
    kv_heads: 2,
    seq_q: 19,
    seq_kv: 27,
    head_dim: 16,
    val_dim: 16,
    visibility: AttentionVisibility::Causal,
};

/// Whichever routine autotune elects, the gradients are the host's.
#[test]
fn the_tuned_backward_pass_matches_the_reference() {
    let device = cubecl::test_device();
    let client = device.client();
    let reference = Reference::new(&client, CASE, half::f16::elem_type_native());
    let gradients = attention_backward(&device, reference.tensors.clone());
    reference.check(&client, gradients, 3e-2);
}
