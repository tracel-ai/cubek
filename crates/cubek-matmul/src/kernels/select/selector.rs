//! How a [`MatmulOpStrategy`] runs a product on a device ([`select`]): what becomes of its packed
//! inputs, and the kernel that reads them. Pure, so the rules are unit-tested against
//! [`Hardware`] fixtures with no GPU. The selector runs first; measuring is one of its answers,
//! and for it the answer is a *table*, not a kernel.
//!
//! # The tables, and the line between them
//!
//! A product races on a **few-rows** table or a **many-rows** one, the [`MatmulTable`]s, whatever its
//! inputs. The few-rows and many-rows tables share no candidate but the unit routine (the few-rows
//! table's floor) and the multi-level routines' naive floor: the two regimes are different kernel bodies, so a single
//! table would be one whose rows mostly decline, paying a compile each to say so. Within a table a
//! float product and a packed one race the same list: each candidate is listed for one kind of
//! product and declines the other before compiling anything, so a key races what its inputs are
//! listed for. A packed product races the settings worth trying on its packed inputs, and the
//! float routines on them decoded first.
//!
//! # The NVFP4 table
//!
//! A third table, [`MatmulTable::Nvfp4`], is reached only by naming
//! [`Nvfp4TensorCores`](crate::kernels::MatmulKernel::Nvfp4TensorCores). Its kernel multiplies two
//! NVFP4 inputs, 4-bit by 4-bit, on the device's block-scaled instruction. Autotune never sends a
//! product there: an NVFP4 product under a policy goes where every other packed product goes. The
//! table races only that kernel's settings, which all give the same numbers.
//!
//! The line is [`FLOAT_FEW_ROWS`] for a float product and
//! [`QUANTIZED_FEW_ROWS`](super::packed::QUANTIZED_FEW_ROWS) for a
//! packed one, each a **rule** rather than a measurement: a staged kernel can be competitive on
//! time at a handful of rows while reading a whole input through a tiling nobody needed, so
//! which table a row count belongs to is decided by what the kernels do. A rule tuned to one
//! device's sweep instead is a number every other device has to live with.

use super::packed::{NVFP4_SCALE_WORD_SPAN, matmul_supported, nvfp4_scheme, quantized_few_rows};
use super::problem::{MatmulProblem, OperandKind};
use super::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
use cubecl::quant::scheme::QuantScheme;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_microkernel::matmul::stored::Operand;
use cubek_std::Hardware;
use cubek_std::matmul_axes::K;
use cubek_std::{NVFP4_BLOCK, block_scaled_nvfp4};

/// How a product runs: what becomes of its packed inputs, and the kernel that reads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatmulSelection {
    /// Float inputs, read by the kernel `choice` gives.
    Float(MatmulChoice),
    /// Packed inputs read as they are stored, by the kernel `choice` gives.
    Packed(MatmulPackedChoice),
    /// Every packed input dequantized into the product's element by `dequantize`, then read by
    /// the float kernel `choice` gives.
    DequantizeThen {
        dequantize: MatmulDequantize,
        choice: MatmulChoice,
    },
}

impl MatmulSelection {
    /// How `strategy` runs a product whose packed inputs the kernels stream, in rows or in tiles:
    /// each kernel reads their layout or declines it at the launch.
    fn packed(
        strategy: MatmulOpStrategy,
        problem: &MatmulProblem,
        hw: &Hardware,
    ) -> Result<Self, Refusal> {
        use MatmulKernel as K;
        use MatmulKernelHeuristic as H;
        use MatmulOpStrategy::*;
        let packed = |choice| Ok(MatmulSelection::Packed(choice));
        let rule = |kernel, otherwise| MatmulPackedChoice::Rule {
            kernel,
            otherwise: Box::new(otherwise),
        };
        let decoded = || {
            MatmulChoice::new(strategy, problem).map(|choice| MatmulSelection::DequantizeThen {
                dequantize: MatmulDequantize::Decode,
                choice,
            })
        };
        // What every narrow candidate counts, which is not the trailing extent.
        let rows = problem.streamed_rows;
        let single_row = rows == Some(1);
        match strategy {
            Autotune => packed(MatmulPackedChoice::Autotune(MatmulTable::packed(problem))),
            // The packed plane routine with the heuristic's own ceiling behind it, since it can still
            // decline a contraction its steps do not close, which no rule here sees.
            Heuristic(H::Tile) if single_row => packed(rule(K::PlaneShuffle, decoded()?)),
            Heuristic(H::Cubek) if single_row => packed(rule(
                K::PlaneShuffle,
                MatmulSelection::Packed(rule(K::CubekAuto, decoded()?)),
            )),
            // Below the `Cubek` ceiling the packed inputs are decoded and run on the tiled routines;
            // at it, the multi-level selection reads them as they are stored.
            Heuristic(H::Tile) | Specific(K::UnitRegisters) => decoded(),
            Heuristic(H::Cubek) => packed(rule(K::CubekAuto, decoded()?)),
            Specific(kernel @ (K::PlaneShuffle | K::CubekAuto)) => {
                packed(MatmulPackedChoice::Pin(kernel))
            }
            Specific(K::Nvfp4TensorCores) if Self::nvfp4_tensor_cores_serve(problem, hw) => {
                packed(MatmulPackedChoice::Pin(K::Nvfp4TensorCores))
            }
            Specific(K::Nvfp4TensorCores) => Err(NVFP4_TENSOR_CORES_UNSERVED),
        }
    }

    /// Whether the 4-bit tensor cores serve this product, on a GPU whose 4-bit instruction takes it:
    /// both inputs in NVFP4 ([`nvfp4_scheme`]), `e2m1` values under a `ue4m3` scale every
    /// [`NVFP4_BLOCK`] of `k` and one `f32` over the whole of it, their words along `k`, `k` whole
    /// words of block scales.
    fn nvfp4_tensor_cores_serve(problem: &MatmulProblem, hw: &Hardware) -> bool {
        let nvfp4 = nvfp4_scheme();
        let format = |scheme: &QuantScheme| {
            (scheme.value, scheme.block_scale().map(|block| block.dtype))
                == (nvfp4.value, nvfp4.block_scale().map(|block| block.dtype))
                && scheme.tensor_scale() == nvfp4.tensor_scale()
        };
        let along_k = |scheme: &QuantScheme, input: MatmulOperand| {
            Operand::axis_of(scheme, input.axes()) == Some(K)
                && Operand::of_scheme(scheme, K, input.axes())
                    .is_some_and(|operand| operand.scale(K) == Some(NVFP4_BLOCK))
        };
        let nvfp4_input = |input: MatmulOperand| match problem.operands.get(input) {
            OperandKind::Quantized(scheme) => format(&scheme) && along_k(&scheme, input),
            OperandKind::Float => false,
        };
        MatmulOperand::INPUTS.into_iter().all(nvfp4_input)
            && problem.k.is_multiple_of(NVFP4_SCALE_WORD_SPAN)
            && block_scaled_nvfp4(hw)
    }
}

/// How the kernel reading float inputs is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulChoice {
    /// The fastest candidate of the table.
    Autotune(MatmulTable),
    /// The kernel a rule names where it serves these operands, the fixed float routine where it declines.
    /// The rules read the shape, not the strides the caller arrived with, and a rule may not fail
    /// a launch over a question it could not ask: whether the plane shuffle binds the strides, or
    /// whether the multi-level selection finds an accelerated path ("no path on this device" comes back as
    /// an error at launch, which no rule can ask in advance).
    Rule(MatmulKernel),
    /// The kernel a pin names, and only it: a decline is the caller's contract broken, so it is
    /// loud.
    Pin(MatmulKernel),
}

impl MatmulChoice {
    /// The kernel `strategy` chooses for float inputs, or packed ones decoded first.
    fn new(strategy: MatmulOpStrategy, problem: &MatmulProblem) -> Result<Self, Refusal> {
        use MatmulKernel as K;
        use MatmulKernelHeuristic as H;
        use MatmulOpStrategy::*;
        match strategy {
            Autotune => Ok(MatmulChoice::Autotune(MatmulTable::new(problem))),
            // No foreign kernel and nothing to measure: the tiled float routines are the whole pool
            // below the `Cubek` ceiling, and which of them serves a call is a question about strides,
            // answered at the launch.
            Heuristic(H::Tile) => Ok(MatmulChoice::Rule(K::PlaneShuffle)),
            // At the ceiling the rules run out: no constant says which staged routine suits a shape, so
            // the answer comes from the selector that owns those routines rather than from a guess.
            Heuristic(H::Cubek) => Ok(MatmulChoice::Rule(K::CubekAuto)),
            Specific(kernel @ (K::UnitRegisters | K::PlaneShuffle | K::CubekAuto)) => {
                Ok(MatmulChoice::Pin(kernel))
            }
            Specific(K::Nvfp4TensorCores) => Err(NVFP4_TENSOR_CORES_UNSERVED),
        }
    }
}

/// How the kernel reading packed inputs as they are stored is chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatmulPackedChoice {
    /// The fastest candidate of the table. On the few-rows table each candidate reads the packed
    /// inputs as they are stored; on the many-rows table the packed staged routine does, and each
    /// float routine runs on them decoded, priced with the decode included, because a caller pays
    /// it every launch. The NVFP4
    /// table only a pin reaches.
    Autotune(MatmulTable),
    /// The kernel a rule names where it serves these operands, and `otherwise` where it declines:
    /// a kernel's plan reads the strides and the scheme's block, and the multi-level selection whether the
    /// device has a path, which the rules could not ask, and a rule may not fail a launch over it.
    Rule {
        kernel: MatmulKernel,
        otherwise: Box<MatmulSelection>,
    },
    /// The kernel a pin names, and only it: a decline is the caller's contract broken, so it is
    /// loud.
    Pin(MatmulKernel),
}

/// How a packed input becomes a float one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulDequantize {
    /// Decoded by the caller's dequantize routine, which places the plane shuffle's planes over
    /// the input, in rows or in tiles. Where the decode declines (a device that commits to no plane width, a
    /// shape its placement does not cut), the framework's dequantize instead: the decode's placement reads
    /// what the rules cannot ask, and a rule may not fail a launch over it.
    Decode,
    /// Dequantized by the framework at the product's element: a scheme no kernel here reads
    /// ([`matmul_supported`]).
    Burn,
}

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
    /// Two NVFP4 inputs multiplied on a GPU's 4-bit tensor cores, at each of [`Nvfp4TensorCores`](crate::kernels::MatmulKernel::Nvfp4TensorCores)'s
    /// settings: what that pin runs, and nothing else.
    Nvfp4,
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

/// How `problem` runs under `strategy` on `hw`.
///
/// `Autotune` measures, on the table the rows go to. `Heuristic` picks by rule, reaching no further
/// than its ceiling: `Tile` stays on the tiled routines, and `Cubek` is the one that may hand a
/// shape to the multi-level selection, which it does wherever these rules have nothing better than a
/// guess. No heuristic measures. A `Specific` names its kernel.
///
/// Float inputs are read by the kernel the strategy chooses. Packed inputs under schemes the
/// kernels stream are read as they are stored by the plane shuffle at one row, by the multi-level
/// selection at the `Cubek` ceiling, and by the kernel a pin names; elsewhere they are decoded first
/// and read by the kernel the strategy chooses for float inputs. Where a packed input's scheme is
/// one no kernel here reads, the framework dequantizes the packed inputs first. Which input is
/// packed never matters.
///
/// # Errors
///
/// A pin that cannot serve the problem: `Nvfp4TensorCores` but on two NVFP4 inputs its GPU's
/// 4-bit instruction takes. The launch panics on it: a pin is a contract with the caller, not a
/// preference.
pub fn select(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
) -> Result<MatmulSelection, Refusal> {
    let mut packed = problem.packed_inputs().peekable();
    if packed.peek().is_none() {
        MatmulChoice::new(strategy, problem).map(MatmulSelection::Float)
    } else if packed.all(|(input, scheme)| matmul_supported(hw, &scheme, input)) {
        MatmulSelection::packed(strategy, problem, hw)
    } else {
        Ok(MatmulSelection::DequantizeThen {
            dequantize: MatmulDequantize::Burn,
            choice: MatmulChoice::new(strategy, problem)?,
        })
    }
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

/// `Nvfp4TensorCores` named for a product it does not serve.
pub const NVFP4_TENSOR_CORES_UNSERVED: Refusal = Refusal::Declined {
    why: "Nvfp4TensorCores multiplies two NVFP4 inputs, their words along k, on a GPU with a 4-bit \
              tensor-core instruction; take autotune or a policy for any other product",
};
