//! Benchmark registry for cubek.

pub use cubek_attention::eval::backward::benchmarks as attention_backward;
pub use cubek_attention::eval::forward::benchmarks as attention;
pub use cubek_convolution::eval::benchmarks as conv2d;
pub use cubek_convolution::eval::benchmarks::depthwise;
pub use cubek_fft::eval::benchmarks as fft;
pub use cubek_interpolate::eval::benchmarks as interpolate;
pub use cubek_matmul::eval::benchmarks::gemm;
pub use cubek_matmul::eval::benchmarks::gemm_cpu;
pub use cubek_matmul::multi_level::eval::gemv;
pub use cubek_matmul::multi_level::eval::quantized_matmul;
pub use cubek_matmul::tiled::eval::gemm_cpu_tiled;
pub use cubek_matmul::tiled::eval::split_k;
pub use cubek_matmul::tiled::eval::tile_quant_stage;
pub use cubek_pool::eval::benchmarks as pool;
pub use cubek_random::eval::benchmarks as random;
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
        &crate::depthwise::Category,
        &crate::fft::Category,
        &crate::gemm::Category,
        &crate::gemm_cpu::Category,
        &crate::gemm_cpu_tiled::Category,
        &crate::gemv::Category,
        &crate::interpolate::Category,
        &crate::interpolate::CpuCategory,
        &crate::memcpy_async::Category,
        &crate::pool::Category,
        &crate::quantized_matmul::Category,
        &crate::random::Category,
        &crate::reduce::Category,
        &crate::split_k::Category,
        &crate::tile_quant_stage::Category,
        &crate::unary::Category,
    ]
}

/// Loop over every (strategy, problem) for `category`, run each at 10 samples,
/// and print the resulting durations using the category's preferred
/// [`cubecl::benchmark::TimingMethod`]. Used by `benches/*.rs` via [`run_bench!`].
///
/// `CUBEK_BENCH_SAMPLES` overrides the sample count (default 10), which matters
/// on a GPU whose clocks idle low (a desktop card at 210MHz of a 3150MHz max,
/// say): a handful of short samples then gets measured while the clocks are
/// still ramping, and identical kernels can differ by the ratio of those two
/// clocks. Raising the count is the difference between a usable number and noise.
pub fn run_category(category: &dyn BenchmarkCategory) {
    use cubecl::benchmark::BenchmarkDurations;

    let samples: usize = std::env::var("CUBEK_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    for problem in category.problems() {
        for strategy in category.strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match category.run(&strategy.id, &problem.id, samples) {
                Ok(samples) => {
                    // Both ceilings, so a row says which one the kernel ran into
                    // rather than only how fast it went.
                    if let Some(compute) = &samples.compute {
                        let achieved_tflops = compute.achieved_ops_per_s / 1e12;
                        match compute.peak_ops_per_s {
                            Some(peak) if peak > 0.0 => {
                                let pct = 100.0 * compute.achieved_ops_per_s / peak;
                                println!("{achieved_tflops:.3} TFLOPS ({pct:.0}% of compute peak)");
                            }
                            _ => println!("{achieved_tflops:.3} TFLOPS (compute peak unavailable)"),
                        }
                    }
                    if let Some(bandwidth) = &samples.bandwidth {
                        let achieved_gb_s = bandwidth.achieved_bytes_per_s / 1e9;
                        match bandwidth.peak_bytes_per_s {
                            Some(peak) if peak > 0.0 => {
                                let pct = 100.0 * bandwidth.achieved_bytes_per_s / peak;
                                println!("{achieved_gb_s:.1} GB/s ({pct:.0}% of memory peak)");
                            }
                            _ => println!("{achieved_gb_s:.1} GB/s (memory peak unavailable)"),
                        }
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
