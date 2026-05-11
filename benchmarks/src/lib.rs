//! Benchmark registry for cubek.
//!
//! The catalogues themselves (problems, strategies, the bench harness, the
//! seeded `Correctness` impls) live in their respective kernel crates under
//! `cubek-<kernel>::benchmarks`, gated behind a `benchmarks` feature. This
//! crate re-exports them with shorter aliases (`attention`, `gemm`, ...) and
//! collects them into [`all()`].

pub use cubek_attention::eval::benchmarks as attention;
pub use cubek_convolution::eval::benchmarks as conv2d;
pub use cubek_fft::eval::benchmarks as fft;
pub use cubek_interpolate::eval::benchmarks as interpolate;
pub use cubek_matmul::eval::benchmarks::gemm;
pub use cubek_matmul::eval::benchmarks::gemv;
pub use cubek_matmul::eval::benchmarks::quantized_matmul;
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
        &crate::contiguous::Category,
        &crate::conv2d::Category,
        &crate::fft::Category,
        &crate::gemm::Category,
        &crate::gemv::Category,
        &crate::interpolate::Category,
        &crate::memcpy_async::Category,
        &crate::quantized_matmul::Category,
        &crate::reduce::Category,
        &crate::unary::Category,
    ]
}

/// Loop over every (strategy, problem) for `category`, run each at 10 samples,
/// and print the resulting durations using the category's preferred
/// [`cubecl::benchmark::TimingMethod`]. Used by `benches/*.rs` via [`run_bench!`].
pub fn run_category(category: &dyn BenchmarkCategory) {
    use cubecl::benchmark::BenchmarkDurations;

    const SAMPLES: usize = 10;

    for problem in category.problems() {
        for strategy in category.strategies() {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match category.run(&strategy.id, &problem.id, SAMPLES) {
                Ok(samples) => {
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
