use benchmarks::fft;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in fft::problems() {
        for strategy in fft::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match fft::run(&strategy.id, &problem.id, 10) {
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
