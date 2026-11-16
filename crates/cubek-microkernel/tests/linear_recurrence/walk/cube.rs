//! The cube walk microkernel alone, under each update rule: a cube's units each hold rows of one
//! value head's state for a run of its columns, through every position.

use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::linear_recurrence::walk::UpdateRule;
use cubek_microkernel::linear_recurrence::walk::cube::CubeWalkMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::ScopeUnderTest;

use crate::linear_recurrence::walk::{F16, F32, Walk, WalkCase};

/// A cube walks every head as the host does, under both update rules at each of its settings,
/// written in `f16` and `f32`: one
/// position and many, one key head and two each read by two value heads, square heads and heads
/// whose value features are wider than their key features.
#[test]
fn a_cube_walks_every_head_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let heads = [
        RecurrenceHeads {
            key_heads: 1,
            heads_per_group: 1,
            key_features: 64,
            value_features: 64,
        },
        RecurrenceHeads {
            key_heads: 2,
            heads_per_group: 2,
            key_features: 32,
            value_features: 128,
        },
    ];
    let cases: Vec<_> = CubeWalkMicrokernel::settings(&hw)
        .into_iter()
        .flat_map(|settings| {
            [UpdateRule::Delta, UpdateRule::Additive]
                .into_iter()
                .flat_map(move |rule| {
                    [F16, F32].into_iter().flat_map(move |element| {
                        heads.into_iter().flat_map(move |heads| {
                            [1, 9].map(|positions| WalkCase {
                                element,
                                rule,
                                heads,
                                positions,
                                settings,
                            })
                        })
                    })
                })
        })
        .collect();
    Walk { hw }.check(cases);
}
