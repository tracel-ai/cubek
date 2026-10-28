//! What each strategy tries, written down: for every strategy and every kind of call, the
//! kernels a launch tries and in what order, one line each, in `plans.txt`.
//!
//! The file is the record a change to the selection is held to: a change that moves a line has
//! changed what runs. Rewrite it with `CUBEK_WRITE_PLANS=1` and read the diff.
//!
//! A line reads:
//! - `name@lead`: a routine at its lead, the first setting it lists.
//! - `name@lead(packed cache)`: the routine reading the packed cache as it lies.
//! - `fastest(table)`: the fastest candidate of an autotune table.
//! - `rebuilt(..)`: the packed cache rebuilt as floats first, by the caller.
//! - `in-order[..]`: the first that serves runs; a list held in a list is written flat, and a
//!   list of one as its piece.
//! - `refused: ..`: the selection refuses the call.

use super::AttentionOpStrategy;
use super::selector::{
    AttentionChoice, AttentionSelection, KernelChoice, KvKind, QueryPositions, select,
};
use crate::kernels::AttentionKernel;

/// Every strategy's plan for every kind of call is the one `plans.txt` records.
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

/// Every strategy against every kind of call, one line each.
fn render_all() -> String {
    let mut out = String::new();
    for (name, queries, cache) in [
        (
            "float cache, one query position",
            QueryPositions::One,
            KvKind::Float,
        ),
        (
            "float cache, many query positions",
            QueryPositions::Many,
            KvKind::Float,
        ),
        (
            "packed cache, one query position",
            QueryPositions::One,
            KvKind::Packed,
        ),
        (
            "packed cache, many query positions",
            QueryPositions::Many,
            KvKind::Packed,
        ),
    ] {
        out.push_str(&format!("# {name}\n"));
        for strategy in AttentionOpStrategy::all() {
            let plan = select(strategy, queries, cache)
                .map(|selection| tried(&selection))
                .unwrap_or_else(|refusal| format!("refused: {refusal}"));
            out.push_str(&format!("{}: {plan}\n", strategy.name()));
        }
        out.push('\n');
    }
    out
}

/// What a launch of `selection` tries, in order.
fn tried(selection: &AttentionSelection) -> String {
    match selection {
        AttentionSelection::Float(choice) => float_tried(choice),
        AttentionSelection::DequantizeThen(choice) => rebuilt(choice),
        AttentionSelection::Packed { read, otherwise } => {
            let read = kernel_tried(read, "(packed cache)");
            match otherwise {
                Some(choice) => in_order([read, rebuilt(choice)]),
                None => read,
            }
        }
    }
}

/// What a launch of `choice` tries over a float cache.
fn float_tried(choice: &AttentionChoice) -> String {
    match choice {
        AttentionChoice::Autotune(QueryPositions::One) => "fastest(decode)".to_string(),
        AttentionChoice::Autotune(QueryPositions::Many) => "fastest(prefill)".to_string(),
        AttentionChoice::Kernel(choice) => kernel_tried(choice, ""),
    }
}

/// The packed cache rebuilt as floats, then `choice` over it.
fn rebuilt(choice: &AttentionChoice) -> String {
    format!("rebuilt({})", float_tried(choice))
}

/// What a launch of `choice` tries, each kernel read as `reads` says: a pin its kernel alone, a
/// rule its kernel and then each one its scope names instead, down to the unit routine.
fn kernel_tried(choice: &KernelChoice, reads: &str) -> String {
    let lead = |kernel: AttentionKernel| format!("{}@lead{reads}", kernel.name());
    match *choice {
        KernelChoice::Pin(kernel) => lead(kernel),
        KernelChoice::Rule(kernel) => in_order(
            core::iter::successors(Some(kernel), AttentionKernel::instead_by_rule).map(lead),
        ),
    }
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
