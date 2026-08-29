//! Generic launch-and-capture-outcome plumbing shared by every kernel-test
//! helper.
//!
//! Kernel launches can fail in two windows: synchronously (the launch closure
//! returns `Err`) or asynchronously when the runtime processes the queued
//! work. An asynchronous failure is not reported by any flush — it lives on
//! the buffers the launch never wrote — so catching it means asking about the
//! launch's own outputs with [`ComputeClient::check`].
//!
//! What comes back is then two different things wearing one shape. A kernel
//! this backend cannot build at this configuration is a skip the policy in
//! [`base`](crate::test_mode::base) may accept; a device fault, an
//! out-of-memory or an IO failure is a defect it must not. Reporting the
//! second as the first is how a broken run reads as a passing one, so the
//! classification is [`ServerError::is_refusal`](cubecl::server::ServerError::is_refusal)'s
//! to make, and not a formatted string's.

use cubecl::{TestRuntime, prelude::ComputeClient, server::Handle};

use crate::{ExecutionOutcome, TestOutcome, ValidationResult};

/// Run `launch` against `client`, returning its [`ExecutionOutcome`]. A
/// failure that surfaces only asynchronously is caught by checking `outputs` —
/// the buffers the launch was going to write, which carry the failure when it
/// never ran.
///
/// No pre-launch check is needed: a failure belongs to the buffers of the
/// launch that failed, so a stale error from an earlier launch cannot be
/// attributed to this one.
///
/// # Panics
///
/// Through the active policy, when the outputs carry a failure that is not a
/// refusal — the run is broken rather than unsupported, and
/// [`ExecutionOutcome`] has no way to say so that a caller could accidentally
/// accept.
#[track_caller]
pub fn launch_and_capture_outcome<F>(
    client: &ComputeClient<TestRuntime>,
    outputs: &[&Handle],
    launch: F,
) -> ExecutionOutcome
where
    F: FnOnce(&ComputeClient<TestRuntime>) -> ExecutionOutcome,
{
    debug_assert!(
        !outputs.is_empty(),
        "a launch with no outputs to check cannot report an asynchronous failure"
    );

    match launch(client) {
        ExecutionOutcome::Executed => match Unrun::of(client, outputs) {
            Some(unrun) => unrun.outcome(),
            None => ExecutionOutcome::Executed,
        },
        other => other,
    }
}

/// Why a launch's outputs cannot be trusted.
enum Unrun {
    /// The backend turned the kernel down — it does not compile here, or it
    /// asked for more resources than the device has. An expected outcome for
    /// a configuration this hardware does not serve.
    Refused(String),
    /// Something went wrong running it: a fault, an out-of-memory, an IO
    /// failure. Never an expected outcome.
    Broken(String),
}

impl Unrun {
    /// Ask whether `outputs` can be trusted, and what stopped them if not.
    ///
    /// `None` when every output checks clean, which is the kernel having run.
    /// The check is one lookup per handle — no read, no barrier.
    fn of(client: &ComputeClient<TestRuntime>, outputs: &[&Handle]) -> Option<Self> {
        let error = client.check(outputs.iter().copied()).err()?;
        let reason = format!("{error:?}");
        Some(match error.is_refusal() {
            true => Self::Refused(reason),
            false => Self::Broken(reason),
        })
    }

    /// The outcome to report, having first failed the test outright if the
    /// run was broken.
    ///
    /// [`ExecutionOutcome`] can only say "did not compile", and every policy
    /// that validates accepts that — so a broken run has to be rejected here,
    /// where the two are still told apart, rather than handed back in a shape
    /// that has lost the difference. A policy for which failing is the point
    /// accepts it and falls through.
    #[track_caller]
    fn outcome(self) -> ExecutionOutcome {
        match self {
            Self::Refused(reason) => ExecutionOutcome::CompileError(reason),
            Self::Broken(reason) => {
                TestOutcome::Validated(ValidationResult::Error(reason.clone())).enforce();
                ExecutionOutcome::CompileError(reason)
            }
        }
    }
}
