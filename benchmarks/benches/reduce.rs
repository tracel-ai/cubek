use benchmarks::reduce;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in reduce::problems() {
        for strategy in reduce::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match reduce::run(&strategy.id, &problem.id, 10) {
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
