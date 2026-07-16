//! Benchmark registry for cubek.

pub use cubek_attention::eval::backward::benchmarks as attention_backward;
pub use cubek_attention::eval::forward::benchmarks as attention;
pub use cubek_convolution::eval::benchmarks as conv2d;
pub use cubek_fft::eval::benchmarks as fft;
pub use cubek_interpolate::eval::benchmarks as interpolate;
pub use cubek_matmul::eval::benchmarks::gemm;
pub use cubek_matmul::eval::benchmarks::gemm_cpu;
pub use cubek_matmul::eval::benchmarks::gemm_cpu_tiled;
pub use cubek_matmul::eval::benchmarks::gemv;
pub use cubek_matmul::eval::benchmarks::quantized_matmul;
pub use cubek_pool::eval::benchmarks as pool;
pub use cubek_reduce::eval::benchmarks as reduce;
pub use cubek_std::eval::benchmarks::contiguous;
pub use cubek_std::eval::benchmarks::memcpy_async;
pub use cubek_std::eval::benchmarks::unary;

pub use cubek_test_utils::{
    BenchmarkCategory, CatalogEntry, Category, Correctness, HostData, ItemDescriptor, RunSamples,
    ValidationResult, compare_host_data_files, read_host_data, write_host_data,
};

/// Every benchmark category compiled into this build of the registry.
pub fn all() -> &'static [&'static dyn BenchmarkCategory] {
    &[
        &crate::attention::Category,
        &crate::attention_backward::Category,
        &crate::contiguous::Category,
        &crate::conv2d::Category,
        &crate::fft::Category,
        &crate::gemm::Category,
        &crate::gemm_cpu::Category,
        &crate::gemm_cpu_tiled::Category,
        &crate::gemv::Category,
        &crate::interpolate::Category,
        &crate::memcpy_async::Category,
        &crate::pool::Category,
        &crate::quantized_matmul::Category,
        &crate::reduce::Category,
        &crate::unary::Category,
    ]
}

/// Loop over every (strategy, problem) for `category`, run each at 10 samples,
/// and print the resulting durations using the category's preferred
/// [`cubecl::benchmark::TimingMethod`]. Used by `benches/*.rs` via [`run_bench!`].
///
/// Three env vars narrow or lengthen a run, which matters on a GPU whose clocks
/// idle low (a desktop card at 210MHz of a 3150MHz max, say): a handful of short
/// samples then gets measured while the clocks are still ramping, and identical
/// kernels can differ by the ratio of those two clocks. Isolating one case and
/// raising the sample count is the difference between a usable number and noise.
///
/// * `CUBEK_BENCH_PROBLEM`  — substring; keep only matching problem ids
/// * `CUBEK_BENCH_STRATEGY` — substring; keep only matching strategy ids
/// * `CUBEK_BENCH_SAMPLES`  — sample count (default 10)
pub fn run_category(category: &dyn BenchmarkCategory) {
    use cubecl::benchmark::BenchmarkDurations;

    let problem_filter = std::env::var("CUBEK_BENCH_PROBLEM").unwrap_or_default();
    let strategy_filter = std::env::var("CUBEK_BENCH_STRATEGY").unwrap_or_default();
    let samples: usize = std::env::var("CUBEK_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    for problem in category.problems() {
        if !problem.id.contains(&problem_filter) {
            continue;
        }
        for strategy in category.strategies() {
            if !strategy.id.contains(&strategy_filter) {
                continue;
            }
            println!("---- {} / {} ----", strategy.label, problem.label);
            match category.run(&strategy.id, &problem.id, samples) {
                Ok(samples) => {
                    if let Some(tflops) = samples.tflops {
                        println!("{tflops:.3} TFLOPS");
                    }
                    let durations = BenchmarkDurations {
                        timing_method: category.timing_method(),
                        durations: samples.durations,
                    };
                    println!("{durations}");
                }
                Err(err) => println!("error: {err}"),
            }
        }
    }
}

/// Generate the `fn main()` for a per-category bench file. Pass the category
/// module name (e.g. `gemm`); the macro resolves to `$crate::gemm::Category`.
#[macro_export]
macro_rules! run_bench {
    ($category:ident) => {
        fn main() {
            $crate::run_category(&$crate::$category::Category);
        }
    };
}
