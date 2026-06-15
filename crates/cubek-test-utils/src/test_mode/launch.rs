//! Generic launch-and-capture-outcome plumbing shared by every kernel-test
//! helper.
//!
//! Kernel launches can fail in two windows: synchronously (the launch closure
//! returns `Err`) or asynchronously when the runtime processes the queued
//! work. Catching the asynchronous case requires an explicit `flush` both
//! before and after the launch.

use cubecl::{
    TestRuntime,
    prelude::ComputeClient,
    server::{self, LaunchError, ServerError},
};

use crate::ExecutionOutcome;

/// Run `launch` against `client`, returning its [`ExecutionOutcome`] after
/// flushing for any compile/launch errors that surface only asynchronously.
///
/// The pre-flush also catches stale errors from a prior launch on the same
/// client — without it, an earlier failure would be attributed to this one.
pub fn launch_and_capture_outcome<F>(
    client: &ComputeClient<TestRuntime>,
    launch: F,
) -> ExecutionOutcome
where
    F: FnOnce(&ComputeClient<TestRuntime>) -> ExecutionOutcome,
{
    let outcome = flush_compile_error(client).unwrap_or_else(|| launch(client));
    match outcome {
        ExecutionOutcome::Executed => {
            flush_compile_error(client).unwrap_or(ExecutionOutcome::Executed)
        }
        other => other,
    }
}

/// Flush `client` and surface any pending compile/launch failure as a
/// [`ExecutionOutcome::CompileError`].
///
/// Returns `None` when the flush is clean (the kernel ran). Other server
/// errors are wrapped as `CompileError` so callers see one uniform shape.
pub fn flush_compile_error(client: &ComputeClient<TestRuntime>) -> Option<ExecutionOutcome> {
    match client.flush() {
        Ok(_) => None,
        Err(ServerError::ServerUnhealthy { errors, .. }) => {
            for error in errors.iter() {
                if let server::ServerError::Launch(LaunchError::TooManyResources(_))
                | server::ServerError::Launch(LaunchError::CompilationError(_)) = error
                {
                    return Some(ExecutionOutcome::CompileError(format!("{errors:?}")));
                }
            }
            None
        }
        Err(err) => Some(ExecutionOutcome::CompileError(format!("{err:?}"))),
    }
}
