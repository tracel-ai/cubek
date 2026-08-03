use core::fmt::Debug;

use cubecl::prelude::*;

use crate::definition::{QRProblem, QRSetupError};

/// How the blueprint for a routine is obtained: either fully specified by the
/// caller (`Forced`) or derived from the problem and the hardware (`Inferred`).
#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: QRRoutine> {
    /// Use this exact blueprint, only deriving the launch settings from it.
    Forced(R::Blueprint),
    /// Derive the blueprint from the problem, the hardware and the given
    /// routine strategy.
    Inferred(R::Strategy),
}

/// A routine adapts a QR algorithm to the hardware it runs on.
///
/// It does not make hard decisions about hardware specifics; instead it reads
/// the device properties and the problem description, and produces:
/// - a [`QRRoutine::Blueprint`]: the minimal comptime settings baked into the
///   generated kernels (a different blueprint triggers a new JIT compilation),
/// - a [`QRRoutine::LaunchSettings`]: the runtime parameters (cube dimensions,
///   cube counts, ...) used to launch those kernels.
pub trait QRRoutine: Debug + Clone + Sized {
    /// Tunable knobs for this routine that do not depend on the hardware.
    type Strategy: Debug + Clone + Default + Send + 'static;
    /// Minimal comptime specialization settings for the kernels.
    type Blueprint: Debug + Clone + Send + 'static;
    /// Runtime launch parameters derived from the problem and the hardware.
    type LaunchSettings: Clone;

    /// Adapt the algorithm to the problem and the hardware, producing the
    /// blueprint for the compiler and the launch settings for the runtime.
    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &QRProblem,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(Self::Blueprint, Self::LaunchSettings), QRSetupError>;
}
