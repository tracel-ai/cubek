//! The unit random microkernel alone: one unit draws its run line after line.

use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::random::{Random, RandomCase, RandomScope};

/// A unit draws every distribution value for value as the host replays it, one and four values a
/// line.
#[test]
fn a_unit_draws_every_distribution_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases = RandomCase::every_distribution(RandomScope::Unit { lines: 64 }, &[1, 4]);
    Random { hw }.check(cases);
}
