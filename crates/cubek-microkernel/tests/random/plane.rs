//! The plane random microkernel alone: every lane of one plane draws the lines
//! `lane + i · lanes` of the plane's run.

use cubek_microkernel::random::plane::PlaneRandomMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::ScopeUnderTest;

use crate::random::{Random, RandomCase, RandomScope};

/// A plane draws every distribution value for value as the host replays it, at each of its
/// settings, one and four values a line.
#[test]
fn a_plane_draws_every_distribution_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = PlaneRandomMicrokernel::settings(&hw)
        .into_iter()
        .flat_map(|settings| RandomCase::every_distribution(RandomScope::Plane(settings), &[1, 4]))
        .collect();
    Random { hw }.check(cases);
}
