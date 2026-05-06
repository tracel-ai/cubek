use benchmarks::interpolate;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in interpolate::problems() {
        for strategy in interpolate::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match interpolate::run(&strategy.id, &problem.id, 10) {
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
