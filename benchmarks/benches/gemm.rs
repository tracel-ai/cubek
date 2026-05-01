use benchmarks::gemm;
use cubecl::benchmark::{BenchmarkDurations, TimingMethod};

fn main() {
    for problem in gemm::problems() {
        for strategy in gemm::strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match gemm::run(&strategy.id, &problem.id, 10) {
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
