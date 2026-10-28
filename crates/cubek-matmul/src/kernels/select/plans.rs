//! What each strategy tries, written down: for every strategy and every kind of product, the
//! kernels a launch tries and in what order, one line each, in `plans.txt`.
//!
//! The file is the record a change to the selection is held to: a change that moves a line has
//! changed what runs. Rewrite it with `CUBEK_WRITE_PLANS=1` and read the diff.
//!
//! A line reads:
//! - `name`: a routine at the first of its settings that serves; `name@default` at its default.
//! - `name(packed as stored)`: the routine reading the packed inputs as they are stored.
//! - `fastest(table)`: the fastest candidate of an autotune table.
//! - `decoded(..)`: every packed input decoded into floats first, by this crate's decode, or by the
//!   framework's dequantize where the decode declines.
//! - `burn-dequantized(..)`: every packed input dequantized by the framework first.
//! - `in-order[..]`: the first that serves runs; a list held in a list is written flat, and a
//!   list of one as its piece.
//! - `refused: ..`: the selection refuses the product.
//!
//! What follows a selection's decline is decided by the caller that launches it, so this module
//! states it here: the fixed routines a float rule falls back to.

use super::packed::{QUANTIZED_FEW_ROWS, nvfp4_scheme};
use super::problem::{MatmulProblem, OperandKind};
use super::selector::{
    FLOAT_FEW_ROWS, MatmulChoice, MatmulDequantize, MatmulPackedChoice, MatmulSelection,
    MatmulTable, select,
};
use super::{MatmulKernel, MatmulOpStrategy};
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_microkernel::matmul::MatmulOperands;
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
            let plan = select(strategy, &product.problem, &product.hw)
                .map(|selection| tried(&selection))
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
    /// Each row count either side of the tables' lines, for float inputs, for packed inputs the
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

/// What a launch of `selection` tries, in order.
fn tried(selection: &MatmulSelection) -> String {
    match selection {
        MatmulSelection::Float(choice) => float_tried(choice),
        MatmulSelection::Packed(choice) => packed_tried(choice),
        MatmulSelection::DequantizeThen { dequantize, choice } => {
            let step = match dequantize {
                MatmulDequantize::Decode => "decoded",
                MatmulDequantize::Burn => "burn-dequantized",
            };
            format!("{step}({})", float_tried(choice))
        }
    }
}

/// What a launch of `choice` tries on float inputs: a rule falls back to the fixed routines, a
/// pin to nothing.
fn float_tried(choice: &MatmulChoice) -> String {
    match choice {
        MatmulChoice::Autotune(table) => fastest(table),
        MatmulChoice::Pin(kernel) => float_kernel(kernel).to_string(),
        MatmulChoice::Rule(kernel) => in_order(
            [
                float_kernel(kernel),
                "cube-staged@default",
                "unit-registers@default",
            ]
            .map(String::from),
        ),
    }
}

/// What a launch of `choice` tries reading the packed inputs as stored.
fn packed_tried(choice: &MatmulPackedChoice) -> String {
    match choice {
        MatmulPackedChoice::Autotune(table) => fastest(table),
        MatmulPackedChoice::Pin(kernel) => packed_kernel(kernel),
        MatmulPackedChoice::Rule { kernel, otherwise } => {
            in_order([packed_kernel(kernel), tried(otherwise)])
        }
    }
}

/// The routine a float launch of `kernel` runs.
fn float_kernel(kernel: &MatmulKernel) -> &'static str {
    match kernel {
        MatmulKernel::UnitRegisters => "unit-registers@default",
        MatmulKernel::PlaneShuffle => "plane-shuffle",
        MatmulKernel::Nvfp4TensorCores => unreachable!("a float product's selection refuses it"),
    }
}

/// What a launch of `kernel` runs reading the packed inputs as stored: the NVFP4 kernel is its
/// own table, whose candidates are its settings.
fn packed_kernel(kernel: &MatmulKernel) -> String {
    match kernel {
        MatmulKernel::PlaneShuffle => "plane-shuffle(packed as stored)".to_string(),
        MatmulKernel::Nvfp4TensorCores => fastest(&MatmulTable::Nvfp4),
        MatmulKernel::UnitRegisters => {
            unreachable!("a packed selection runs it on the decoded inputs")
        }
    }
}

fn fastest(table: &MatmulTable) -> String {
    let name = match table {
        MatmulTable::FewRows => "few-rows",
        MatmulTable::ManyRows => "many-rows",
        MatmulTable::Nvfp4 => "nvfp4",
    };
    format!("fastest({name})")
}

/// `pieces` tried in order, a list among them written flat and a list of one as its piece:
/// trying one piece in order is trying that piece.
fn in_order(pieces: impl IntoIterator<Item = String>) -> String {
    let flat: Vec<String> = pieces
        .into_iter()
        .map(|piece| match piece.strip_prefix("in-order[") {
            Some(list) => list.strip_suffix(']').expect("a list closes").to_string(),
            None => piece,
        })
        .collect();
    match flat.as_slice() {
        [one] => one.clone(),
        _ => format!("in-order[{}]", flat.join(", ")),
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
