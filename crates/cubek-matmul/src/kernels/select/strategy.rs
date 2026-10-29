//! **What a caller can ask the matmul family for.**
//!
//! Every variant but `Specific` is a policy; `Specific` is a pin. Which
//! kernel one resolves to is [`selector`](super::selector)'s.

use serde::{Deserialize, Serialize};

/// How a matmul picks its kernel.
///
/// Three answers, and each says *how* the choice is made rather than naming a
/// mechanism: measure everything, follow the rules up to a ceiling, or run one
/// named kernel. A [`Specific`](Self::Specific) that cannot serve the problem
/// panics at the launch rather than falling back, because the caller asked for
/// that kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum MatmulOpStrategy {
    /// Pick by measurement: every kernel that can serve the problem joins an
    /// autotune table and the fastest runs, per problem and per device.
    ///
    /// The default. The rules cannot know where the
    /// crossover between a streaming kernel and a staged one falls on a device
    /// nobody has run yet — it moves in *sign*, not just in size — so the
    /// answer is measured.
    ///
    /// Only kernels are measured. There is no framework op among the
    /// candidates: a framework's `matmul` runs an autotune of its own, so an autotune
    /// holding it would be a tuner inside a tuner whose winning candidate names
    /// a wrapper. The routines are named and launched directly instead, which
    /// is what makes every candidate a kernel.
    ///
    /// Which kernels join, and in what order, is the autotune's own answer,
    /// because its candidate set is built once per process and no
    /// strategy passed per call could change it.
    #[default]
    Autotune,
    /// Pick by rule, reaching no further than the named architecture: the routines without a
    /// tuning round, which is what the correctness suites hold the rules to.
    Heuristic(MatmulKernelHeuristic),
    /// Run exactly this kernel, for measuring it or holding it to a reference.
    Specific(MatmulKernel),
}

/// How far from the routines in [`kernels`](crate::kernels) a
/// [`Heuristic`](MatmulOpStrategy::Heuristic) may reach: a **ceiling**, named as attention names
/// its own, of which the matmul family has one.
///
/// Nothing here measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulKernelHeuristic {
    /// The routines in [`kernels`](crate::kernels) only: a branch runs one, and another of them
    /// where it declines (the plane shuffle its fixed fallback, the packed plane shuffle the
    /// dequantized float pair). The one foreign kernel it reaches is the framework's
    /// dequantize, where a quantized input is materialized.
    Tile,
}

/// The kernels themselves, named for what they are.
///
/// This is the vocabulary a [`Specific`](MatmulOpStrategy::Specific) pin
/// speaks, so a benchmark can drive one kernel *through* a framework's matmul
/// instead of reaching past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulKernel {
    /// The float routine that asks nothing of the hardware,
    /// `UnitRegistersMatmulRoutine`, at its default setting: every unit owns a
    /// block of the output, so it serves any row count on any device, and is
    /// the pin a reference names where a float product of any shape must run.
    /// It binds its operands through the tile DSL's projection, so it reads an operand
    /// stored in row-first tiles as well as in rows, and takes the elements
    /// every routine takes.
    UnitRegisters,
    /// The memory-bound plane routine: few enough rows to read the rhs about
    /// once, every row held against each read while a plane's units split the
    /// contraction. `PlaneShuffleMatmulRoutine`, on a float rhs or on a
    /// packed one read as stored. Runs the first of its settings that serves
    /// the product.
    // A strategy written down under the pin's earlier name still reads as this one.
    #[serde(alias = "PlaneFold")]
    PlaneShuffle,
    /// Two packed inputs multiplied on the GPU's block-scaled tensor cores, both read as they are
    /// stored: NVFP4, MXFP4, or any pair of formats the device has an instruction for. The inputs
    /// a caller rounded to a format are its choice of what the product computes, so no policy runs
    /// it. Named here for a benchmark or a test; two packed inputs a block-scaled instruction of
    /// the GPU takes, only.
    // A strategy written down under the pin's earlier name still reads as this one.
    #[serde(alias = "Nvfp4TensorCores")]
    BlockScaledTensorCores,
}

impl MatmulOpStrategy {
    /// Every strategy: the policies, then a pin of every kernel.
    pub fn all() -> Vec<Self> {
        let heuristics = MatmulKernelHeuristic::all()
            .into_iter()
            .map(Self::Heuristic);
        let pins = MatmulKernel::all().into_iter().map(Self::Specific);
        core::iter::once(Self::Autotune)
            .chain(heuristics)
            .chain(pins)
            .collect()
    }

    /// Whether this strategy lets the caller pick the kernel, rather than
    /// naming one.
    ///
    /// The split is what every caller holding a strategy branches on: a policy
    /// may be served by whatever serves the problem — a fused replacement, the
    /// reference path on a device with no kernels — while a pin is a contract
    /// with a benchmark row or a capture, and substituting for it would measure
    /// or record a kernel nobody asked for.
    pub fn is_policy(&self) -> bool {
        !matches!(self, Self::Specific(_))
    }
}

impl MatmulKernelHeuristic {
    /// Every ceiling, nearest first.
    pub fn all() -> Vec<Self> {
        vec![Self::Tile]
    }
}

impl MatmulKernel {
    /// Every kernel a pin names.
    pub fn all() -> Vec<Self> {
        vec![
            Self::UnitRegisters,
            Self::PlaneShuffle,
            Self::BlockScaledTensorCores,
        ]
    }
}
