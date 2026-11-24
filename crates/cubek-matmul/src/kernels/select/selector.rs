//! How a [`MatmulOpStrategy`] runs a product on a device ([`select`]): what becomes of its packed
//! inputs, and the kernel that reads them. Pure, so the rules are unit-tested against
//! [`Hardware`] fixtures with no GPU. The selector runs first; measuring is one of its answers,
//! and for it the answer is an autotune, not a kernel.

use super::packed::{block_scaled_instruction_reads, matmul_supported};
use super::problem::MatmulProblem;
use super::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
use crate::kernels::plan::{Decoded, FrameworkDequantized, MatmulPlan, RoutinePlan};
use crate::kernels::tune::{MatmulAutotune, MatmulAutotuneGroups};
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutine,
    UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings,
};
use cubecl::tile::launch::Decline;
use cubek_std::Hardware;
use cubek_std::plan::{InOrder, Then};

/// The autotune `strategy`'s plan for `problem` on `hw` is, where its plan is one autotune alone:
/// what a caller that launches the autotune's winner itself asks the autotune about
/// ([`MatmulAutotune::fastest_routine`]). `None` where the plan is anything else, a packed input
/// the framework dequantizes first among them.
pub fn measured_autotune(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
) -> Option<MatmulAutotune> {
    if strategy != MatmulOpStrategy::Autotune {
        return None;
    }
    match Inputs::of(problem, hw) {
        Inputs::Float => Some(MatmulAutotune::new(problem)),
        Inputs::Streamed => Some(MatmulAutotune::packed(problem)),
        Inputs::Unread => None,
    }
}

/// How `problem` runs under `strategy` on `hw`: the plan its launch tries, the autotunes it names
/// shipping their groups at `groups`.
///
/// `Autotune` measures, on the autotune the rows go to. `Heuristic` picks by rule, on the tiled
/// routines, and never measures. A `Specific` names its kernel.
///
/// Float inputs are read by the kernel the strategy chooses. Packed inputs under schemes the
/// kernels stream are read as they are stored by the plane shuffle at one row, by the kernel a
/// pin names, and by autotune's candidates, the block-scaled tensor cores among them past a decode's
/// rows; elsewhere they are decoded first and read by the kernel the strategy chooses for float
/// inputs. Where a packed input's scheme is one no kernel here reads, the framework dequantizes the
/// packed inputs first. Which input is packed never matters.
///
/// # Errors
///
/// A pin that cannot serve the problem: `BlockScaledTensorCores` but on two packed inputs a
/// block-scaled instruction of its GPU reads. The launch panics on it: a pin is a contract with the
/// caller, not a preference.
pub fn select(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
    groups: MatmulAutotuneGroups,
) -> Result<MatmulPlan, Decline> {
    match Inputs::of(problem, hw) {
        Inputs::Float => float_plan(strategy, problem, groups),
        Inputs::Streamed => packed_plan(strategy, problem, hw, groups),
        Inputs::Unread => {
            let plan = float_plan(strategy, problem, groups)?;
            Ok(Box::new(Then::new(FrameworkDequantized, plan)))
        }
    }
}

/// What a product's inputs are, as the selection tells them apart.
enum Inputs {
    /// Every input is float.
    Float,
    /// Every packed input is under a scheme the kernels stream ([`matmul_supported`]).
    Streamed,
    /// A packed input is under a scheme no kernel here reads.
    Unread,
}

impl Inputs {
    fn of(problem: &MatmulProblem, hw: &Hardware) -> Self {
        let mut packed = problem.packed_inputs().peekable();
        if packed.peek().is_none() {
            Self::Float
        } else if packed.all(|(input, scheme)| matmul_supported(hw, &scheme, input)) {
            Self::Streamed
        } else {
            Self::Unread
        }
    }
}

/// How `strategy` runs float inputs, or packed ones decoded first: an autotune, a rule's kernel and
/// then the fixed routines that serve what it declines, or a pin's kernel alone.
///
/// The rules read the shape, not the strides the caller arrived with, and a rule may not fail a
/// launch over a question it could not ask: whether the plane shuffle binds the strides. So a rule
/// ends on the routine that asks nothing of the device. A pin's decline is the caller's contract
/// broken, so it is loud.
fn float_plan(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    groups: MatmulAutotuneGroups,
) -> Result<MatmulPlan, Decline> {
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    match strategy {
        Autotune => Ok(MatmulAutotune::new(problem).plan(groups)),
        // Nothing to measure: the tiled float routines are the whole pool, and which of them
        // serves a call is a question about strides, answered at the launch.
        Heuristic(H::Tile) => Ok(Box::new(InOrder::new(vec![
            plane_shuffle(),
            Box::new(RoutinePlan::at(
                CubeStagedMatmulRoutine,
                "cube-staged@default",
                CubeStagedMatmulRoutineSettings::DEFAULT,
            )),
            unit_registers_at_default(),
        ]))),
        Specific(K::UnitRegisters) => Ok(unit_registers_at_default()),
        Specific(K::PlaneShuffle) => Ok(plane_shuffle()),
        Specific(K::BlockScaledTensorCores) => Err(BLOCK_SCALED_TENSOR_CORES_UNSERVED),
    }
}

/// How `strategy` runs a product whose packed inputs the kernels stream, in rows or in tiles:
/// each kernel reads their layout or declines it at the launch, the packed inputs then decoded
/// for the float plan a rule falls back to.
fn packed_plan(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
    groups: MatmulAutotuneGroups,
) -> Result<MatmulPlan, Decline> {
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    let decoded = || -> Result<MatmulPlan, Decline> {
        let plan = float_plan(strategy, problem, groups)?;
        Ok(Box::new(Then::new(Decoded, plan)))
    };
    // What every narrow candidate counts, which is not the trailing extent.
    let single_row = problem.streamed_rows == Some(1);
    match strategy {
        Autotune => Ok(MatmulAutotune::packed(problem).plan(groups)),
        // The packed plane routine with the heuristic's own fallback behind it, since it can still
        // decline a contraction its steps do not close, which no rule here sees.
        Heuristic(H::Tile) if single_row => Ok(Box::new(InOrder::new(vec![
            packed_plane_shuffle(),
            decoded()?,
        ]))),
        // Past one row the packed inputs are decoded and run on the tiled routines.
        Heuristic(H::Tile) | Specific(K::UnitRegisters) => decoded(),
        Specific(K::PlaneShuffle) => Ok(packed_plane_shuffle()),
        Specific(K::BlockScaledTensorCores)
            if block_scaled_instruction_reads(hw, &problem.operands) =>
        {
            Ok(MatmulAutotune::BlockScaled.plan(groups))
        }
        Specific(K::BlockScaledTensorCores) => Err(BLOCK_SCALED_TENSOR_CORES_UNSERVED),
    }
}

/// `BlockScaledTensorCores` named for a product it does not serve.
pub const BLOCK_SCALED_TENSOR_CORES_UNSERVED: Decline = Decline::Unserved {
    why: "BlockScaledTensorCores multiplies two packed inputs, their words along k, on a GPU with a \
              block-scaled tensor-core instruction of their format; take autotune or a policy for \
              any other product",
};

/// The memory-bound plane routine, at the first of its settings that serves.
fn plane_shuffle() -> MatmulPlan {
    Box::new(RoutinePlan::first_served(
        PlaneShuffleMatmulRoutine,
        "plane-shuffle",
    ))
}

/// The plane routine reading the packed inputs as they are stored.
fn packed_plane_shuffle() -> MatmulPlan {
    Box::new(RoutinePlan::first_served(
        PlaneShuffleMatmulRoutine,
        "plane-shuffle(packed as stored)",
    ))
}

/// The routine that asks nothing of the device, at its default: what serves any float product.
fn unit_registers_at_default() -> MatmulPlan {
    Box::new(RoutinePlan::at(
        UnitRegistersMatmulRoutine,
        "unit-registers@default",
        UnitRegistersMatmulRoutineSettings::DEFAULT,
    ))
}
