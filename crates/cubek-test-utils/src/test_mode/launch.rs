//! Generic launch-and-capture-outcome plumbing shared by every kernel-test
//! helper.
//!
//! Kernel launches can fail in two windows: synchronously (the launch closure
//! returns `Err`) or asynchronously when the runtime processes the queued
//! work. An asynchronous failure is not reported by any flush — it lives on
//! the buffers the launch never wrote — so catching it means asking about the
//! launch's own outputs with [`ComputeClient::check`].

use cubecl::{
    TestRuntime,
    prelude::ComputeClient,
    server::{Handle, ServerError},
};

use crate::ExecutionOutcome;

/// Run `launch` against `client`, returning its [`ExecutionOutcome`]. A
/// compile/launch failure that surfaces only asynchronously is caught by
/// checking `outputs` — the buffers the launch was going to write, which
/// carry the failure when it never ran.
///
/// No pre-launch check is needed any more: a failure belongs to the buffers
/// of the launch that failed, so a stale error from an earlier launch cannot
/// be attributed to this one.
pub fn launch_and_capture_outcome<F>(
    client: &ComputeClient<TestRuntime>,
    outputs: &[&Handle],
    launch: F,
) -> ExecutionOutcome
where
    F: FnOnce(&ComputeClient<TestRuntime>) -> ExecutionOutcome,
{
    match launch(client) {
        ExecutionOutcome::Executed => {
            check_compile_error(client, outputs).unwrap_or(ExecutionOutcome::Executed)
        }
        other => other,
    }
}

/// Ask whether `outputs` can be trusted, surfacing the failure a launch left
/// on them as an [`ExecutionOutcome::CompileError`].
///
/// Returns `None` when every output checks clean (the kernel ran). The check
/// is one lookup per handle — no read, no barrier — and every failure kind is
/// wrapped as `CompileError` so callers see one uniform shape, exactly as the
/// flush-based classification did.
pub fn check_compile_error(
    client: &ComputeClient<TestRuntime>,
    outputs: &[&Handle],
) -> Option<ExecutionOutcome> {
    for handle in outputs {
        if let Err(err) = client.check(handle) {
            return Some(compile_error(&err));
        }
    }
    None
}

fn compile_error(err: &ServerError) -> ExecutionOutcome {
    ExecutionOutcome::CompileError(format!("{err:?}"))
}
