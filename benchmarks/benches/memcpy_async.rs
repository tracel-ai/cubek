use benchmarks::memcpy_async;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in memcpy_async::problems() {
        for strategy in memcpy_async::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match memcpy_async::run(&strategy.id, &problem.id, 10) {
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
