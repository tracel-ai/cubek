use benchmarks::gemv;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in gemv::problems() {
        for strategy in gemv::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match gemv::run(&strategy.id, &problem.id, 10) {
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
