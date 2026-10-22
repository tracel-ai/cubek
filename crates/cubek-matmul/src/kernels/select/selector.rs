//! How a [`MatmulOpStrategy`] runs a product on a device ([`select`]): what becomes of its weight,
//! and the kernel that reads it. Pure, so the rules are unit-tested against
//! [`Hardware`] fixtures with no GPU. The selector runs first; measuring is one of its answers,
//! and for it the answer is a *table*, not a kernel.
//!
//! # The tables, and the line between them
//!
//! A product races on a **gemv** table or a **gemm** one, the [`MatmulTable`]s, whatever its
//! weight. The few-rows and many-rows tables share no candidate but the unit routine (the few-rows
//! table's floor) and cubek's naive floor: the two regimes are different kernel bodies, so a single
//! table would be one whose rows mostly decline, paying a compile each to say so. Within a table a
//! float weight and a packed one race the same list: each candidate is listed for one kind of
//! weight and declines the other before compiling anything, so a key races what its weight is
//! listed for. A packed weight races the settings worth trying against it, and the float routines
//! on the weight decoded first.
//!
//! # The NVFP4 table
//!
//! A third table, [`MatmulTable::Nvfp4`], is reached only by naming
//! [`Nvfp4TensorCores`](crate::kernels::MatmulKernel::Nvfp4TensorCores). Its kernel rounds the
//! activation to NVFP4 and multiplies 4-bit by 4-bit, a different product from the one an NVFP4
//! weight asks for: the weight is packed and the activation stays in 16-bit. Autotune never sends
//! a product there, since a table races settings of one product and never chooses which product
//! runs; an NVFP4 weight under a policy goes where every other packed weight goes. The table races
//! only that kernel's settings, which all give the same numbers.
//!
//! The line is [`FLOAT_FEW_ROWS`] for a float weight and
//! [`QUANTIZED_FEW_ROWS`](super::packed::QUANTIZED_FEW_ROWS) for a
//! packed one, each a **rule** rather than a measurement: a staged kernel can be competitive on
//! time at a handful of rows while reading the whole weight through a tiling nobody needed, so
//! which table a row count belongs to is decided by what the kernels do. A rule tuned to one
//! device's sweep instead is a number every other device has to live with.

use super::packed::{
    NVFP4_SCALE_WORD_SPAN, matmul_supported, nvfp4_activation_scheme, quantized_few_rows,
};
use super::problem::{MatmulProblem, OperandKind};
use super::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
use cubecl::quant::scheme::QuantScheme;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::stored::Operand;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, N};
use cubek_std::{NVFP4_BLOCK, block_scaled_nvfp4};

/// How a product runs: what becomes of its weight, and the kernel that reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatmulSelection {
    /// A float weight, read by the kernel `choice` gives.
    Float(MatmulChoice),
    /// A packed weight read as it is stored, by the kernel `choice` gives.
    Packed(MatmulPackedChoice),
    /// A packed weight dequantized into the product's element by `dequantize`, a packed activation
    /// too, then read by the float kernel `choice` gives.
    DequantizeThen {
        dequantize: MatmulDequantize,
        choice: MatmulChoice,
    },
}

/// How the kernel reading a float weight is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulChoice {
    /// The fastest candidate of the table.
    Autotune(MatmulTable),
    /// The kernel a rule names where it serves these operands, the fixed gemm where it declines.
    /// The rules read the shape, not the strides the caller arrived with, and a rule may not fail
    /// a launch over a question it could not ask: whether the plane shuffle binds the strides, or
    /// whether cubek's selection finds an accelerated path ("no path on this device" comes back as
    /// an error at launch, which no rule can ask in advance).
    Rule(MatmulKernel),
    /// The kernel a pin names, and only it: a decline is the caller's contract broken, so it is
    /// loud.
    Pin(MatmulKernel),
}

/// How the kernel reading a packed weight as it is stored is chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatmulPackedChoice {
    /// The fastest candidate of the table. On the few-rows table each candidate reads the weight as it
    /// is stored; on the many-rows table the packed staged gemm does, and each float routine runs on it
    /// decoded, priced with its decode included, because a caller pays it every launch. The NVFP4
    /// table only a pin reaches.
    Autotune(MatmulTable),
    /// The kernel a rule names where it serves these operands, and `otherwise` where it declines:
    /// a kernel's plan reads the strides and the scheme's block, and cubek's selection whether the
    /// device has a path, which the rules could not ask, and a rule may not fail a launch over it.
    Rule {
        kernel: MatmulKernel,
        otherwise: Box<MatmulSelection>,
    },
    /// The kernel a pin names, and only it: a decline is the caller's contract broken, so it is
    /// loud.
    Pin(MatmulKernel),
}

/// How a packed weight becomes a float one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulDequantize {
    /// Decoded by the caller's dequantize routine, which places the plane shuffle's planes over
    /// the weight, in rows or in tiles. Where the decode declines (a device that commits to no plane width, a
    /// shape its placement does not cut), burn's dequantize instead: the decode's placement reads
    /// what the rules cannot ask, and a rule may not fail a launch over it.
    Decode,
    /// Dequantized by burn at the activation's element: a scheme no kernel here reads
    /// ([`matmul_supported`]).
    Burn,
}

/// The autotune tables a product races on, split by the rows a gemv would stream: each holds the
/// candidates for a float weight and for a packed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulTable {
    /// A handful of rows, each read once: the gemv routines, against a packed weight at the
    /// settings worth trying against it, reading it as stored.
    FewRows,
    /// Everything past the gemv row bound: the gemm routines, and against a packed weight the
    /// packed staged gemm reading it as stored and the float routines on it decoded first.
    ManyRows,
    /// The activation rounded to NVFP4 and multiplied against an NVFP4 weight on a GPU's 4-bit
    /// tensor cores, at each of [`Nvfp4TensorCores`](crate::kernels::MatmulKernel::Nvfp4TensorCores)'s
    /// settings: what that pin runs, and nothing else.
    Nvfp4,
}

impl MatmulTable {
    /// Which table a float product goes to, by [`float_few_rows`], so no width reaches a table whose
    /// every candidate declines it. A packed weight decoded into a float one first is a float
    /// product here.
    pub fn new(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(float_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }

    /// Which table a product against a packed weight read as stored goes to, by the rows alone
    /// ([`quantized_few_rows`]): the few-rows table's candidates decline what they do not close,
    /// and its floor, the weight decoded, answers there.
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
/// than its ceiling: `Tile` stays on this crate's kernels, and `Cubek` is the one that may hand a
/// shape to cubek's own selector, which it does wherever these rules have nothing better than a
/// guess. No heuristic measures. A `Specific` names its kernel.
///
/// A float weight is read by the kernel the strategy chooses. A packed weight under a scheme the
/// kernels stream is read as it is stored by the plane shuffle at one row, by cubek at the `Cubek`
/// ceiling, and by the kernel a pin names; elsewhere it is decoded first and read by the kernel the
/// strategy chooses for a float weight. A scheme no kernel here reads is dequantized by burn
/// first, under every strategy but the `Nvfp4TensorCores` pin. A quantized activation against a
/// packed weight takes the selection a float one would: the packed routines read it as stored, and
/// every other kernel is handed it dequantized.
///
/// # Errors
///
/// A pin that cannot serve the problem: `Nvfp4TensorCores` but on an NVFP4 weight its GPU's 4-bit
/// instruction takes. A quantized activation against a float weight. The launch panics on either:
/// a pin is a contract with the caller, not a preference.
pub fn select(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    hw: &Hardware,
) -> Result<MatmulSelection, Refusal> {
    match (problem.lhs, problem.rhs) {
        (OperandKind::Quantized(_), OperandKind::Float) => Err(QUANTIZED_LHS_FLOAT_RHS),
        (_, OperandKind::Float) => float_choice(strategy, problem).map(MatmulSelection::Float),
        (_, OperandKind::Quantized(scheme)) if matmul_supported(hw, &scheme) => {
            packed_selection(strategy, problem, &scheme, hw)
        }
        (_, OperandKind::Quantized(_)) => Ok(MatmulSelection::DequantizeThen {
            dequantize: MatmulDequantize::Burn,
            choice: float_choice(strategy, problem)?,
        }),
    }
}

/// The most rows a float product goes to the few-rows table with: the rows every float gemv candidate
/// holds whole, each one more accumulator line per cell of its register block (the plane shuffle's
/// `MAX_ROWS` is held to it by a test). Past it, a body that reuses the weight across rows is the
/// better shape anyway.
pub const FLOAT_FEW_ROWS: usize = 8;

/// Whether a float product of `rows` rows goes to the few-rows table: from one row to
/// [`FLOAT_FEW_ROWS`]. A float product of no rows goes to the many-rows table, which has a candidate
/// for it.
pub fn float_few_rows(rows: usize) -> bool {
    (1..=FLOAT_FEW_ROWS).contains(&rows)
}

/// A quantized activation against a float weight: no product reads one, and none is asked for,
/// since a model quantizes an activation to multiply a weight it quantized too.
pub const QUANTIZED_LHS_FLOAT_RHS: Refusal = Refusal::Declined {
    why: "a quantized activation multiplies a quantized weight; this one is float",
};

/// `Nvfp4TensorCores` named for a product it does not serve.
pub const NO_NVFP4_WEIGHT: Refusal = Refusal::Declined {
    why: "Nvfp4TensorCores multiplies an NVFP4 weight, its words along k, on a GPU with a 4-bit \
              tensor-core instruction; take autotune or a policy for any other product",
};

/// The kernel `strategy` chooses for a float weight, or one a packed weight is decoded into.
fn float_choice(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
) -> Result<MatmulChoice, Refusal> {
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    match strategy {
        Autotune => Ok(MatmulChoice::Autotune(MatmulTable::new(problem))),
        // No foreign kernel and nothing to measure: this crate's float routines are the whole pool
        // below the cubek ceiling, and which of them serves a call is a question about strides,
        // answered at the launch.
        Heuristic(H::Tile) => Ok(MatmulChoice::Rule(K::PlaneShuffle)),
        // At the ceiling the rules run out: no constant says which staged GEMM suits a shape, so
        // the answer comes from the selector that owns those routines rather than from a guess.
        Heuristic(H::Cubek) => Ok(MatmulChoice::Rule(K::CubekAuto)),
        Specific(kernel @ (K::UnitRegisters | K::PlaneShuffle | K::CubekAuto)) => {
            Ok(MatmulChoice::Pin(kernel))
        }
        Specific(K::Nvfp4TensorCores) => Err(NO_NVFP4_WEIGHT),
    }
}

/// How `strategy` runs a packed weight under `scheme`, which the kernels stream, in rows or in
/// tiles: each kernel reads its layout or declines it at the launch.
fn packed_selection(
    strategy: MatmulOpStrategy,
    problem: &MatmulProblem,
    scheme: &QuantScheme,
    hw: &Hardware,
) -> Result<MatmulSelection, Refusal> {
    use MatmulKernel as K;
    use MatmulKernelHeuristic as H;
    use MatmulOpStrategy::*;
    let packed = |choice| Ok(MatmulSelection::Packed(choice));
    let rule = |kernel, otherwise| MatmulPackedChoice::Rule {
        kernel,
        otherwise: Box::new(otherwise),
    };
    let decoded = || {
        float_choice(strategy, problem).map(|choice| MatmulSelection::DequantizeThen {
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
        // Below the cubek ceiling the weight is decoded and run on this crate's kernels; at it,
        // cubek reads the weight as it is stored.
        Heuristic(H::Tile) | Specific(K::UnitRegisters) => decoded(),
        Heuristic(H::Cubek) => packed(rule(K::CubekAuto, decoded()?)),
        Specific(kernel @ (K::PlaneShuffle | K::CubekAuto)) => {
            packed(MatmulPackedChoice::Pin(kernel))
        }
        Specific(K::Nvfp4TensorCores) if nvfp4_tensor_cores_serve(problem, scheme, hw) => {
            packed(MatmulPackedChoice::Pin(K::Nvfp4TensorCores))
        }
        Specific(K::Nvfp4TensorCores) => Err(NO_NVFP4_WEIGHT),
    }
}

/// Whether the 4-bit tensor cores serve this product, on a GPU whose 4-bit instruction takes it:
/// a weight packed as the activation is quantized for it ([`nvfp4_activation_scheme`]), `e2m1`
/// values under a `ue4m3` scale every [`NVFP4_BLOCK`] of `k` and one `f32` over the whole of it,
/// its words along `k`, `k` whole words of block scales.
fn nvfp4_tensor_cores_serve(problem: &MatmulProblem, scheme: &QuantScheme, hw: &Hardware) -> bool {
    let activation = nvfp4_activation_scheme();
    let format = (scheme.value, scheme.block_scale().map(|block| block.dtype))
        == (
            activation.value,
            activation.block_scale().map(|block| block.dtype),
        )
        && scheme.tensor_scale() == activation.tensor_scale();
    let along_k = Operand::axis_of(scheme, [K, N]) == Some(K)
        && Operand::of_scheme(scheme, K, [K, N])
            .is_some_and(|operand| operand.scale(K) == Some(NVFP4_BLOCK));
    format && along_k && problem.k.is_multiple_of(NVFP4_SCALE_WORD_SPAN) && block_scaled_nvfp4(hw)
}
