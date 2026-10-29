//! How a [`MatmulOpStrategy`] runs a product on a device ([`select`]): what becomes of its packed
//! inputs, and the kernel that reads them. Pure, so the rules are unit-tested against
//! [`Hardware`] fixtures with no GPU. The selector runs first; measuring is one of its answers,
//! and for it the answer is a *table*, not a kernel.
//!
//! # The tables, and the line between them
//!
//! A product races on a **few-rows** table or a **many-rows** one, the [`MatmulTable`]s, whatever its
//! inputs. The few-rows and many-rows tables share no candidate but the unit routine, their floor:
//! the two regimes are different kernel bodies, so a single table would be one whose rows mostly
//! decline, paying a compile each to say so. Within a table a
//! float product and a packed one race the same list: each candidate is listed for one kind of
//! product and declines the other before compiling anything, so a key races what its inputs are
//! listed for. A packed product races the settings worth trying on its packed inputs, and the
//! float routines on them decoded first.
//!
//! # The block-scaled table
//!
//! A third table, [`MatmulTable::BlockScaled`], is reached only by naming
//! [`BlockScaledTensorCores`](crate::kernels::MatmulKernel::BlockScaledTensorCores). Its kernel
//! multiplies two packed inputs as they are stored on the device's block-scaled instruction of
//! their format: NVFP4's 4-bit by 4-bit, or any other the device has. Autotune never sends a
//! product there: a packed product under a policy goes where every other packed product goes. The
//! table races only that kernel's settings, which all give the same numbers.
//!
//! The line is [`FLOAT_FEW_ROWS`] for a float product and
//! [`QUANTIZED_FEW_ROWS`](super::packed::QUANTIZED_FEW_ROWS) for a
//! packed one, each a **rule** rather than a measurement: a staged kernel can be competitive on
//! time at a handful of rows while reading a whole input through a tiling nobody needed, so
//! which table a row count belongs to is decided by what the kernels do. A rule tuned to one
//! device's sweep instead is a number every other device has to live with.

use super::packed::{matmul_supported, quantized_few_rows};
use super::problem::{MatmulProblem, OperandKind};
use super::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
use crate::kernels::plan::{Decoded, FrameworkDequantized, MatmulPlan, RoutinePlan};
use crate::kernels::tune::MatmulTableGroups;
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutine,
    UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_std::block_scaled_for;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_axes::K;
use cubek_std::plan::{InOrder, Then};

/// The autotune tables a product races on, split by the rows a routine that does not stage would stream: each holds the
/// candidates for a float product and for a packed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulTable {
    /// A handful of rows, each read once: the routines that do not stage, on packed inputs at the
    /// settings worth trying on them, reading them as stored.
    FewRows,
    /// Everything past the few-rows bound: the staged routines, and on packed inputs the packed
    /// staged routine reading them as stored and the float routines on them decoded first.
    ManyRows,
    /// Two packed inputs multiplied on a GPU's block-scaled tensor cores, at each of
    /// [`BlockScaledTensorCores`](crate::kernels::MatmulKernel::BlockScaledTensorCores)'s
    /// settings: what that pin runs, and nothing else.
    BlockScaled,
}

impl MatmulTable {
    /// Which table a float product goes to, by [`float_few_rows`], so no width reaches a table whose
    /// every candidate declines it. Packed inputs decoded first make a float product here.
    pub fn new(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(float_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }

    /// Which table a product reading packed inputs as stored goes to, by the rows alone
    /// ([`quantized_few_rows`]): the few-rows table's candidates decline what they do not close,
    /// and its floor, the packed inputs decoded, answers there.
    pub fn packed(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(quantized_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }
}

/// How `problem` runs under `strategy` on `hw`: the plan its launch tries, the tables it names
/// shipping their groups at `groups`.
///
/// `Autotune` measures, on the table the rows go to. `Heuristic` picks by rule, on the tiled
/// routines, and never measures. A `Specific` names its kernel.
///
/// Float inputs are read by the kernel the strategy chooses. Packed inputs under schemes the
/// kernels stream are read as they are stored by the plane shuffle at one row and by the kernel a
/// pin names; elsewhere they are decoded first
/// and read by the kernel the strategy chooses for float inputs. Where a packed input's scheme is
/// one no kernel here reads, the framework dequantizes the packed inputs first. Which input is
/// packed never matters.
///
/// # Errors
///
/// A pin that cannot serve the problem: `BlockScaledTensorCores` but on two NVFP4 inputs its GPU's
/// 4-bit instruction takes. The launch panics on it: a pin is a contract with the caller, not a
/// preference.
/// The table `strategy`'s plan for `problem` on `hw` is, where its plan is one table alone: what a
/// caller that launches the table's winner itself asks the table about
/// ([`MatmulTable::fastest_routine`]). `None` where the plan is anything else, a packed input the
/// framework dequantizes first among them.
pub fn measured_table(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
) -> Option<MatmulTable> {
    if strategy != MatmulOpStrategy::Autotune {
        return None;
    }
    let mut packed = problem.packed_inputs().peekable();
    if packed.peek().is_none() {
        Some(MatmulTable::new(problem))
    } else if packed.all(|(input, scheme)| matmul_supported(hw, &scheme, input)) {
        Some(MatmulTable::packed(problem))
    } else {
        None
    }
}

pub fn select(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
    groups: MatmulTableGroups,
) -> Result<MatmulPlan, Refusal> {
    let mut packed = problem.packed_inputs().peekable();
    if packed.peek().is_none() {
        float_plan(strategy, problem, groups)
    } else if packed.all(|(input, scheme)| matmul_supported(hw, &scheme, input)) {
        packed_plan(strategy, problem, hw, groups)
    } else {
        let plan = float_plan(strategy, problem, groups)?;
        Ok(Box::new(Then::new(FrameworkDequantized, plan)))
    }
}

/// How `strategy` runs float inputs, or packed ones decoded first: a table, a rule's kernel and then
/// the fixed routines that serve what it declines, or a pin's kernel alone.
///
/// The rules read the shape, not the strides the caller arrived with, and a rule may not fail a
/// launch over a question it could not ask: whether the plane shuffle binds the strides. So a rule
/// ends on the routine that asks nothing of the device. A pin's decline is the caller's contract
/// broken, so it is loud.
fn float_plan(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    groups: MatmulTableGroups,
) -> Result<MatmulPlan, Refusal> {
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    match strategy {
        Autotune => Ok(MatmulTable::new(problem).plan(groups)),
        // Nothing to measure: the tiled float routines are the whole pool, and which of them
        // serves a call is a question about strides, answered at the launch.
        Heuristic(H::Tile) => Ok(Box::new(InOrder::new(vec![
            plane_shuffle(),
            cube_staged_at_default(),
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
    groups: MatmulTableGroups,
) -> Result<MatmulPlan, Refusal> {
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    let decoded = || -> Result<MatmulPlan, Refusal> {
        let plan = float_plan(strategy, problem, groups)?;
        Ok(Box::new(Then::new(Decoded, plan)))
    };
    // What every narrow candidate counts, which is not the trailing extent.
    let single_row = problem.streamed_rows == Some(1);
    match strategy {
        Autotune => Ok(MatmulTable::packed(problem).plan(groups)),
        // The packed plane routine with the heuristic's own fallback behind it, since it can still
        // decline a contraction its steps do not close, which no rule here sees.
        Heuristic(H::Tile) if single_row => Ok(Box::new(InOrder::new(vec![
            packed_plane_shuffle(),
            decoded()?,
        ]))),
        // Past one row the packed inputs are decoded and run on the tiled routines.
        Heuristic(H::Tile) | Specific(K::UnitRegisters) => decoded(),
        Specific(K::PlaneShuffle) => Ok(packed_plane_shuffle()),
        Specific(K::BlockScaledTensorCores) if block_scaled_tensor_cores_serve(problem, hw) => {
            Ok(MatmulTable::BlockScaled.plan(groups))
        }
        Specific(K::BlockScaledTensorCores) => Err(BLOCK_SCALED_TENSOR_CORES_UNSERVED),
    }
}

/// Whether the block-scaled tensor cores serve this product, on a GPU whose block-scaled
/// instruction takes it: both inputs packed with their words along `k` under block scales along
/// `k` ([`block_scaled_for`]), a step of the instruction a word of each input's block scales, and
/// `k` whole such words. A per-tensor scale above the blocks, or none, is the product's to apply.
fn block_scaled_tensor_cores_serve(problem: &MatmulProblem, hw: &Hardware) -> bool {
    let input = |input: MatmulOperand| match problem.operands.get(input) {
        OperandKind::Quantized(scheme) if Operand::axis_of(&scheme, input.axes()) == Some(K) => {
            let operand = Operand::of_scheme(&scheme, K, input.axes())?;
            let block = operand.scale(K)?;
            Some((scheme, block, block * operand.scales_per_word()))
        }
        OperandKind::Quantized(_) | OperandKind::Float => None,
    };
    let (Some((lhs, lhs_block, lhs_word)), Some((rhs, rhs_block, rhs_word))) =
        (input(MatmulOperand::Lhs), input(MatmulOperand::Rhs))
    else {
        return false;
    };
    // The instruction takes one word of each input's scales a step.
    lhs_word == rhs_word
        && problem.k.is_multiple_of(lhs_word)
        && block_scaled_for(hw, (&lhs, lhs_block), (&rhs, rhs_block))
            .any(|instruction| instruction.shape.k as usize == lhs_word)
}

/// The most rows a float product goes to the few-rows table with: the rows every float few-rows candidate
/// holds whole, each one more accumulator line per cell of its register block (the plane shuffle's
/// `MAX_ROWS` is held to it by a test). Past it, a body that reuses each column's values across rows is the
/// better shape anyway.
pub const FLOAT_FEW_ROWS: usize = 8;

/// Whether a float product of `rows` rows goes to the few-rows table: from one row to
/// [`FLOAT_FEW_ROWS`]. A float product of no rows goes to the many-rows table, which has a candidate
/// for it.
pub fn float_few_rows(rows: usize) -> bool {
    (1..=FLOAT_FEW_ROWS).contains(&rows)
}

/// `BlockScaledTensorCores` named for a product it does not serve.
pub const BLOCK_SCALED_TENSOR_CORES_UNSERVED: Refusal = Refusal::Declined {
    why: "BlockScaledTensorCores multiplies two packed inputs, their words along k, on a GPU with a \
              block-scaled tensor-core instruction of their format whose step is a word of their \
              scales; take autotune or a policy for any other product",
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

/// The staged routine at its default: a rule's first fallback.
fn cube_staged_at_default() -> MatmulPlan {
    Box::new(RoutinePlan::at(
        CubeStagedMatmulRoutine,
        "cube-staged@default",
        CubeStagedMatmulRoutineSettings::DEFAULT,
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
