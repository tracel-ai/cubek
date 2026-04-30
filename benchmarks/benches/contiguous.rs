use benchmarks::contiguous;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in contiguous::problems() {
        for strategy in contiguous::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match contiguous::run(&strategy.id, &problem.id, 10) {
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
