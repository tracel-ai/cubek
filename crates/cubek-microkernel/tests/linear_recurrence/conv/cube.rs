//! The cube q|k-normed conv microkernel alone: a cube's units each a line of one key head's
//! channels at one position, the head's queries and keys normed through shared memory.

use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::linear_recurrence::conv::{Conv, ConvCase, F16, F32};

/// A cube convolves and norms every head as the host does, in `f16` and `f32`: one key head and
/// two each read by two value heads, filters of two taps and of four, one position and several.
#[test]
fn a_cube_convolves_every_head_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let heads = [
        RecurrenceHeads {
            key_heads: 1,
            heads_per_group: 1,
            key_features: 16,
            value_features: 16,
        },
        RecurrenceHeads {
            key_heads: 2,
            heads_per_group: 2,
            key_features: 16,
            value_features: 32,
        },
    ];
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|element| {
            heads.into_iter().flat_map(move |heads| {
                [(2, 1), (4, 5)].map(|(taps, positions)| ConvCase {
                    element,
                    heads,
                    taps,
                    positions,
                })
            })
        })
        .collect();
    Conv { hw }.check(cases);
}
