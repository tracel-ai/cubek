//! The cube interpolate microkernel alone: each block of a cube's walk, its units each a run of
//! channels of one output position, contracting the window's taps against their weights.

use cubek_microkernel::testing::interpolate::WindowAxis;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::interpolate::{F16, F32, Interpolate, InterpolateCase};

/// A cube sums every window's weighted taps as the host does: windows that tile the input, that
/// overlap and are padded, and whose taps are spread apart, square and not, a unit one channel a
/// step and four, over one block of the plane and over three.
#[test]
fn a_cube_sums_every_window_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let windows = [
        [WindowAxis::new(2, 2, 1, 0), WindowAxis::new(2, 2, 1, 0)],
        [WindowAxis::new(3, 2, 1, 1), WindowAxis::new(3, 1, 1, 1)],
        [WindowAxis::new(2, 1, 2, 0), WindowAxis::new(3, 2, 2, 2)],
    ];
    let cases: Vec<_> = [F16, F32]
        .into_iter()
        .flat_map(|element| {
            windows.into_iter().flat_map(move |windows| {
                [(1, 1), (4, 3)].map(|(channel_block, walks)| InterpolateCase {
                    element,
                    windows,
                    input: [9, 7],
                    channel_block,
                    walks,
                })
            })
        })
        .collect();
    Interpolate { hw }.check(cases);
}
