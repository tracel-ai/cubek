use benchmarks::unary;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in unary::problems() {
        for strategy in unary::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match unary::run(&strategy.id, &problem.id, 10) {
                Ok(samples) => {
                    let durations = BenchmarkDurations {
                        timing_method: TimingMethod::Device,
                        durations: samples.durations,
                    };
                    println!("{durations}");
                }
                Err(err) => println!("error: {err}"),
            }
        }
    }
}
