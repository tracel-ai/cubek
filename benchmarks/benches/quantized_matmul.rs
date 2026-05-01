use benchmarks::quantized_matmul;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in quantized_matmul::problems() {
        for strategy in quantized_matmul::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match quantized_matmul::run(&strategy.id, &problem.id, 10) {
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
