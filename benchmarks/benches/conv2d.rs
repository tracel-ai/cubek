use benchmarks::conv2d;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in conv2d::problems() {
        for strategy in conv2d::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match conv2d::run(&strategy.id, &problem.id, 10) {
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
