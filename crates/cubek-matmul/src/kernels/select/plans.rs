//! What each strategy tries, written down: for every strategy and every kind of product, the
//! kernels a launch tries and in what order, one line each, in `plans.txt`.
//!
//! The file is the record a change to the selection is held to: a change that moves a line has
//! changed what runs. Rewrite it with `CUBEK_WRITE_PLANS=1` and read the diff.
//!
//! A line reads:
//! - `name`: a routine at the first of its settings that serves; `name@default` at its default.
//! - `name(packed as stored)`: the routine reading the packed inputs as they are stored.
//! - `fastest(autotune)`: the fastest candidate of an autotune.
//! - `decoded(..)`: every packed input decoded into floats first, by this crate's decode, or by the
//!   framework's dequantize where the decode declines.
//! - `framework-dequantized(..)`: every packed input dequantized by the framework first.
//! - `in-order[..]`: the first that serves runs; a list held in a list is written flat, and a
//!   list of one as its piece.
//! - `refused: ..`: the selection refuses the product.

use super::MatmulOpStrategy;
use super::packed::nvfp4_scheme;
use super::problem::{MatmulProblem, OperandKind};
use super::selector::select;
use crate::kernels::tune::{FLOAT_FEW_ROWS, MatmulAutotuneGroups, QUANTIZED_FEW_ROWS};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_std::plan::Plan;
use cubek_std::{Hardware, devices};

/// Every strategy's plan for every kind of product is the one `plans.txt` records.
///
/// A line that moves is a kernel that runs where it did not, or stops running where it did: the
/// diff is the change to review, so a change to the selection rewrites the file on purpose.
#[test]
fn every_plan_is_the_one_written_down() {
    let written = render_all();
    if std::env::var_os("CUBEK_WRITE_PLANS").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels/select/plans.txt");
        std::fs::write(path, &written).expect("plans.txt is writable");
        return;
    }
    assert_eq!(
        written,
        include_str!("plans.txt"),
        "a plan changed; rewrite plans.txt with CUBEK_WRITE_PLANS=1 and review the diff"
    );
}

/// Every strategy against every kind of product, one line each.
fn render_all() -> String {
    let mut out = String::new();
    for product in Product::all() {
        out.push_str(&format!("# {}\n", product.name));
        for strategy in MatmulOpStrategy::all() {
            let groups = MatmulAutotuneGroups::default();
            let plan = select(strategy, &product.problem, &product.hw, groups)
                .map(|plan| plan.tree().to_string())
                .unwrap_or_else(|refusal| format!("refused: {refusal}"));
            out.push_str(&format!("{strategy:?}: {plan}\n"));
        }
        out.push('\n');
    }
    out
}

/// One kind of product the selection tells apart, on the device it is asked about.
struct Product {
    name: &'static str,
    problem: MatmulProblem,
    hw: Hardware,
}

impl Product {
    /// Each row count either side of the autotunes' lines, for float inputs, for packed inputs the
    /// kernels read, for packed inputs none reads, and for two NVFP4 inputs on a GPU with the
    /// 4-bit instruction and on one without.
    fn all() -> Vec<Self> {
        let h100 = devices::h100;
        let blackwell = || devices::offering_nvfp4(devices::h100());
        let float = |rows| float_operands(rows);
        let q4 = |rows| packed_rhs(rows, q4());
        let unread = |rows| packed_rhs(rows, per_tensor());
        let nvfp4 = |rows| both_nvfp4(rows);
        vec![
            Self::new("float, 1 row", float(1), h100()),
            Self::new("float, the few-rows line", float(FLOAT_FEW_ROWS), h100()),
            Self::new(
                "float, past the few-rows line",
                float(FLOAT_FEW_ROWS + 1),
                h100(),
            ),
            Self::new("packed rhs, 1 row", q4(1), h100()),
            Self::new("packed rhs, 2 rows", q4(2), h100()),
            Self::new(
                "packed rhs, the few-rows line",
                q4(QUANTIZED_FEW_ROWS),
                h100(),
            ),
            Self::new(
                "packed rhs, past the few-rows line",
                q4(QUANTIZED_FEW_ROWS + 1),
                h100(),
            ),
            Self::new(
                "rhs packed under a scheme no kernel reads, 1 row",
                unread(1),
                h100(),
            ),
            Self::new("two NVFP4 inputs, 1 row, 4-bit GPU", nvfp4(1), blackwell()),
            Self::new(
                "two NVFP4 inputs, 512 rows, 4-bit GPU",
                nvfp4(512),
                blackwell(),
            ),
            Self::new(
                "two NVFP4 inputs, 1 row, no 4-bit instruction",
                nvfp4(1),
                h100(),
            ),
        ]
    }

    fn new(name: &'static str, problem: MatmulProblem, hw: Hardware) -> Self {
        Self { name, problem, hw }
    }
}

/// `rows` float rows against a float `[k, n]` rhs.
fn float_operands(rows: usize) -> MatmulProblem {
    product(rows, OperandKind::Float, OperandKind::Float)
}

/// `rows` float rows against a `[k, n]` rhs packed under `scheme`.
fn packed_rhs(rows: usize, scheme: QuantScheme) -> MatmulProblem {
    product(rows, OperandKind::Float, OperandKind::Quantized(scheme))
}

/// Two NVFP4 inputs, their words and blocks along `k`.
fn both_nvfp4(rows: usize) -> MatmulProblem {
    let along_k = QuantScheme::default()
        .with_value(QuantValue::E2M1)
        .per_block([16, 1], ScaleDtype::UE4M3)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(1));
    product(
        rows,
        OperandKind::Quantized(nvfp4_scheme()),
        OperandKind::Quantized(along_k),
    )
}

fn product(rows: usize, lhs: OperandKind, rhs: OperandKind) -> MatmulProblem {
    MatmulProblem {
        m: rows,
        k: 2048,
        n: 8192,
        batches: 1,
        streamed_rows: Some(rows),
        operands: MatmulOperands {
            lhs,
            rhs,
            out: OperandKind::Float,
        },
    }
}

/// 4-bit values under 32-wide f32 block scales: a scheme the kernels stream.
fn q4() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .per_block([32], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
}

/// Per-tensor scales: a scheme no kernel here reads.
fn per_tensor() -> QuantScheme {
    QuantScheme::default().with_value(QuantValue::Q8S)
}
