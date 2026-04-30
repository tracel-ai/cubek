use benchmarks::attention;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in attention::problems() {
        for strategy in attention::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match attention::run(&strategy.id, &problem.id, 10) {
                Ok(samples) => {
                    let durations = BenchmarkDurations {
                        timing_method: TimingMethod::System,
                        durations: samples.durations,
                    };
                    println!("{durations}");
                }
                Err(err) => println!("error: {err}"),
            }
        }
    }
}
