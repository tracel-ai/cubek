//! What each strategy tries, written down: for every strategy and every kind of call, the
//! kernels a launch tries and in what order, one line each, in `plans.txt`.
//!
//! The file is the record a change to the selection is held to: a change that moves a line has
//! changed what runs. Rewrite it with `CUBEK_WRITE_PLANS=1` and read the diff.
//!
//! A line reads:
//! - `name@lead`: a routine at its lead, the first setting it lists.
//! - `name@lead(packed cache)`: the routine reading the packed cache as it lies.
//! - `fastest(autotune)`: the fastest candidate of an autotune.
//! - `rebuilt(..)`: the packed cache rebuilt as floats first, by the caller.
//! - `in-order[..]`: the first that serves runs; a list held in a list is written flat, and a
//!   list of one as its piece.
//! - `refused: ..`: the selection refuses the call.

use super::AttentionOpStrategy;
use super::problem::{KvKind, QueryPositions};
use super::selector::select;
use crate::kernels::tune::AttentionAutotuneGroups;
use cubek_std::plan::Plan;

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
            let plan = select(strategy, queries, cache, AttentionAutotuneGroups::default())
                .map(|plan| plan.tree().to_string())
                .unwrap_or_else(|refusal| format!("refused: {refusal}"));
            out.push_str(&format!("{}: {plan}\n", strategy.name()));
        }
        out.push('\n');
    }
    out
}
