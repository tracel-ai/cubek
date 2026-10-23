//! **What a caller can ask attention for.**
//!
//! Every variant but `Specific` is a policy; `Specific` is a pin. Which
//! kernel one resolves to is [`selector`](super::selector)'s.

use crate::kernels::{
    AttentionRoutine, AttentionTensors, CubeAttentionKernel, HypercubeAttentionKernel,
    UnitAttentionRoutine,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use serde::{Deserialize, Serialize};

/// How an attention call picks its kernel.
///
/// Three answers, and each says *how* the choice is made rather than naming a
/// mechanism: measure everything, follow the rules up to a ceiling, or run one
/// named kernel. A [`Specific`](Self::Specific) that cannot serve the problem
/// panics at the launch rather than falling back, because the caller asked for
/// that kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AttentionOpStrategy {
    /// Pick by measurement. Two autotunes: one for calls of many query positions,
    /// where the tiled routines race the flash routines ([`forward`](crate::forward)), and one
    /// for a single position, where they race each other.
    ///
    /// Every candidate of the single position's autotune launches the same kernel
    /// at every step of a cache, its attended length read on device: the
    /// graph-capture precondition, which the list keeps rather than the
    /// measurement. A chunk against a partly filled cache is measured without
    /// the flash op, for a different reason: its causal mask is not
    /// bottom-right aligned on that shape, so the autotune would be measuring two
    /// different functions.
    ///
    /// The default.
    #[default]
    Autotune,
    /// Pick by rule, reaching no further than the named architecture.
    Heuristic(AttentionKernelHeuristic),
    /// Run exactly this kernel, for measuring it or holding it to a reference.
    Specific(AttentionKernel),
}

/// How far from the tiled routines a
/// [`Heuristic`](AttentionOpStrategy::Heuristic) may reach.
///
/// A **ceiling**, not an exclusive choice, as in the matmul family.
///
/// [`Cubek`](Self::Cubek) and [`Autotune`](AttentionOpStrategy::Autotune)
/// resolve identically while every flash candidate is measured: "may
/// reach the flash routines" and "measure everything" then name the same set.
/// They part company once a candidate is left out of autotune, and stating
/// both keeps the vocabulary the same as matmul's rather than encoding the
/// family's kernel count into the type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionKernelHeuristic {
    /// The tiled routines throughout, by rule ([`select`](super::selector::select)). Nothing
    /// here reaches the flash op, and nothing here measures.
    Tile,
    /// The same rules, with the prompt's routines measured against
    /// the flash routines ([`forward`](crate::forward)).
    Cubek,
}

/// The kernels themselves, named for what they are: the unit routine, or a cube or a hypercube
/// routine, each scope naming its own.
///
/// This is the vocabulary a [`Specific`](AttentionOpStrategy::Specific) pin speaks. The flash
/// routines ([`forward`](crate::forward)) are not here: they are reachable as measured candidates,
/// and a caller that wants one unconditionally launches it directly, outside this family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionKernel {
    /// [`UnitAttentionRoutine`] at its lead setting, at any query length, over a float cache:
    /// every unit owning queries of one head, on any device.
    Unit,
    /// A cube routine.
    Cube(CubeAttentionKernel),
    /// A hypercube routine.
    Hypercube(HypercubeAttentionKernel),
}

impl AttentionOpStrategy {
    /// Every strategy, in declaration order, each nested kernel and ceiling spelled out.
    pub fn all() -> Vec<Self> {
        let heuristics = AttentionKernelHeuristic::all()
            .into_iter()
            .map(Self::Heuristic);
        let kernels = AttentionKernel::all().into_iter().map(Self::Specific);
        [Self::Autotune]
            .into_iter()
            .chain(heuristics)
            .chain(kernels)
            .collect()
    }

    /// Whether this strategy lets the caller's policy pick the kernel, rather than naming one. See
    /// [`MatmulOpStrategy::is_policy`](cubek_matmul::kernels::select::MatmulOpStrategy::is_policy).
    pub fn is_policy(&self) -> bool {
        !matches!(self, Self::Specific(_))
    }

    /// What a report calls this strategy, in kebab-case.
    ///
    /// **The one place these names are spelled.** A profile row's variant and a
    /// routing sweep's row name are two surfaces onto the same choice, and two
    /// literal tables would let them drift into disagreeing about which
    /// strategy a run took.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Autotune => "autotune",
            Self::Heuristic(heuristic) => heuristic.name(),
            Self::Specific(kernel) => kernel.name(),
        }
    }

    /// Whether this strategy settles a choice by measuring.
    ///
    /// [`Autotune`](Self::Autotune) does by name, and
    /// [`Cubek`](AttentionKernelHeuristic::Cubek) does because "may reach the flash
    /// routines" is the same set as "measure everything" while every flash
    /// candidate is measured. [`Tile`](AttentionKernelHeuristic::Tile) does not,
    /// and that is the whole of what it means: the tiled routines by rule, at
    /// their lead settings, with nothing timed.
    pub(crate) fn measures(&self) -> bool {
        matches!(
            self,
            Self::Autotune | Self::Heuristic(AttentionKernelHeuristic::Cubek)
        )
    }
}

impl AttentionKernelHeuristic {
    /// Every ceiling, in declaration order.
    pub fn all() -> Vec<Self> {
        vec![Self::Tile, Self::Cubek]
    }

    /// What a report calls this ceiling, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tile => "tile",
            Self::Cubek => "cubek",
        }
    }
}

impl AttentionKernel {
    /// Whether this kernel reads a packed cache as it lies: the hypercube shuffles, either merge.
    pub(crate) fn reads_packed(self) -> bool {
        matches!(
            self,
            AttentionKernel::Hypercube(
                HypercubeAttentionKernel::LastArrivalShuffle
                    | HypercubeAttentionKernel::SecondLaunchShuffle
            )
        )
    }

    /// Every kernel, in declaration order, each scope's routines spelled out.
    pub fn all() -> Vec<Self> {
        let cubes = CubeAttentionKernel::all().into_iter().map(Self::Cube);
        let hypercubes = HypercubeAttentionKernel::all()
            .into_iter()
            .map(Self::Hypercube);
        [Self::Unit]
            .into_iter()
            .chain(cubes)
            .chain(hypercubes)
            .collect()
    }

    /// What a rule that named this kernel runs where it declines on geometry the rules cannot
    /// see: what its scope names instead; nothing after the unit routine, whose lead serves every
    /// problem.
    pub fn instead_by_rule(&self) -> Option<Self> {
        match self {
            Self::Unit => None,
            Self::Cube(cube) => Some(cube.instead_by_rule()),
            Self::Hypercube(hypercube) => Some(hypercube.instead_by_rule()),
        }
    }

    /// What a report calls this kernel, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unit => "unit-attention",
            Self::Cube(cube) => cube.name(),
            Self::Hypercube(hypercube) => hypercube.name(),
        }
    }

    /// The kernel's routine at its lead on `tensors` ([`AttentionRoutine::launch_lead_on`]): what a
    /// pin of it runs.
    ///
    /// # Errors
    ///
    /// Why the routine declines this problem on this device.
    pub fn launch_lead_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
    ) -> Result<TensorHandle, Refusal> {
        match self {
            Self::Unit => UnitAttentionRoutine.launch_lead_on(hw, tensors),
            Self::Cube(cube) => cube.launch_lead_on(hw, tensors),
            Self::Hypercube(hypercube) => hypercube.launch_lead_on(hw, tensors),
        }
    }

    /// The kernel's lead on `tensors` by rule: where it declines on geometry the rules cannot
    /// see, the kernel it names instead ([`instead_by_rule`](Self::instead_by_rule)), down to the
    /// unit routine's lead.
    ///
    /// # Errors
    ///
    /// Why the unit routine's lead declines too: operands no routine serves.
    pub fn launch_by_rule_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
    ) -> Result<TensorHandle, Refusal> {
        let mut kernel = *self;
        loop {
            match kernel.launch_lead_on(hw, tensors.clone()) {
                Ok(out) => return Ok(out),
                Err(why) => match kernel.instead_by_rule() {
                    Some(instead) => kernel = instead,
                    None => return Err(why),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::select::*;

    /// Everything that is not a pin lets the caller's policy choose; a `Specific`
    /// names the kernel a benchmark row or a capture asked for and must not be
    /// substituted.
    #[test]
    fn the_policies_are_the_strategies_that_name_no_kernel() {
        let policies: Vec<_> = AttentionOpStrategy::all()
            .into_iter()
            .filter(AttentionOpStrategy::is_policy)
            .collect();
        assert_eq!(
            policies,
            vec![
                AttentionOpStrategy::Autotune,
                AttentionOpStrategy::Heuristic(AttentionKernelHeuristic::Tile),
                AttentionOpStrategy::Heuristic(AttentionKernelHeuristic::Cubek),
            ],
        );
    }

    /// Autotune is what a caller naming nothing gets, as in the matmul family.
    #[test]
    fn the_default_is_autotune() {
        assert_eq!(
            AttentionOpStrategy::default(),
            AttentionOpStrategy::Autotune
        );
    }
}
