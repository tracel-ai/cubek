//! Thin shim over the registry. Runs every (strategy, problem) pair so that
//! `cargo bench --bench attention` keeps working.

use benchmarks::attention;

fn main() {
    for problem in attention::problems() {
        for strategy in attention::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match attention::run(strategy.id, problem.id, 10) {
                Ok(samples) => {
                    let mut ns: Vec<u128> =
                        samples.durations.iter().map(|d| d.as_nanos()).collect();
                    ns.sort_unstable();
                    let median = ns[ns.len() / 2];
                    println!("median: {:.3} ms", median as f64 / 1.0e6);
                }
                Err(err) => println!("error: {err}"),
            }
        }
    }
}
