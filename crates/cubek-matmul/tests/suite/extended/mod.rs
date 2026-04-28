//! Extended tier: hand-written forced-blueprint tests covering harder or
//! niche cases — per-routine TilingScheme sweep, alt shapes, non-default
//! layouts, hypercube / swizzle / specialization / partition-buffering knobs,
//! and quantization.

use cubecl::{TestRuntime, client::ComputeClient};
use cubek_matmul::{definition::MatmulProblem, launch::test_only::TestStrategy};

use crate::suite::launcher_strategy::run;

mod common;

mod advanced;
mod alt_shapes;
mod layouts;
mod quantization;
mod tiling_scheme;

/// Test the correctness of a [`TestStrategy`] (test-only routines) against
/// the CPU reference. Kept separate from [`test_matmul_strategy`] so the
/// public `Strategy` enum stays lean.
#[allow(unused)]
pub fn test_matmul_test_strategy(
    client: ComputeClient<TestRuntime>,
    problem: MatmulProblem,
    strategy: TestStrategy,
) {
    crate::suite::launcher_strategy::run(client, problem, move |client, lhs, rhs, out, dtypes| {
        strategy.launch_ref(client, lhs, rhs, out, dtypes)
    });
}
