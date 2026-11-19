//! The expert-stack layouts timed in turn rather than one after another
//! ([`interleaved`](benchmarks::expert_stacks::interleaved)), one table a problem: each layout's
//! median, tenth percentile and fastest call, and its median against the padded rows a stack kept
//! in rows holds on CUDA.
//!
//! `CUBEK_BENCH_PROBLEMS` filters the problems as the other benches do; `CUBEK_BENCH_SAMPLES` is
//! the rounds (default 200).

use std::time::Duration;

use benchmarks::expert_stacks::{StackLayout, interleaved, problems, strategies};

fn main() {
    let rounds = std::env::var("CUBEK_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let filter: Vec<String> = std::env::var("CUBEK_BENCH_PROBLEMS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    let layouts: Vec<(String, StackLayout)> = strategies()
        .into_iter()
        .map(|entry| (entry.label, entry.value))
        .collect();
    let kinds: Vec<StackLayout> = layouts.iter().map(|(_, layout)| *layout).collect();
    for problem in problems() {
        if !filter.is_empty() && !filter.iter().any(|want| problem.id.contains(want.as_str())) {
            continue;
        }
        println!("\n## {} ({rounds} rounds)\n", problem.label);
        println!("| layout | median | p10 | min | median vs padded |");
        println!("|---|---:|---:|---:|---:|");
        let timed = interleaved(&problem.value, &kinds, rounds);
        let median_of = |layout| {
            timed
                .iter()
                .find(|(l, _)| *l == layout)
                .and_then(|(_, d)| d.as_ref().ok().map(|d| percentile(d, 50)))
        };
        let padded = median_of(StackLayout::PaddedRows);
        for ((label, _), (_, durations)) in layouts.iter().zip(&timed) {
            match durations {
                Ok(d) => {
                    let median = percentile(d, 50);
                    let ratio = padded
                        .map(|p| format!("{:.2}×", median.as_secs_f64() / p.as_secs_f64()))
                        .unwrap_or_default();
                    println!(
                        "| {label} | {} | {} | {} | {ratio} |",
                        micros(median),
                        micros(percentile(d, 10)),
                        micros(percentile(d, 0)),
                    );
                }
                Err(why) => {
                    let why: String = why.chars().take(140).collect();
                    println!("| {label} | — | — | — | refused: {why} |");
                }
            }
        }
    }
}

/// The `p`th percentile of `durations`.
fn percentile(durations: &[Duration], p: usize) -> Duration {
    let mut sorted = durations.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() - 1) * p / 100]
}

fn micros(d: Duration) -> String {
    format!("{:.1} µs", d.as_secs_f64() * 1e6)
}
