//! Benchmark registry for cubek.

pub mod attention;
pub mod contiguous;
pub mod conv2d;
pub mod fft;
pub mod gemm;
pub mod gemv;
pub mod interpolate;
pub mod memcpy_async;
pub mod quantized_matmul;
pub mod reduce;
pub mod registry;
pub mod unary;

pub use registry::{
    BenchmarkCategory, CatalogEntry, Category, Correctness, ItemDescriptor, RunSamples, all,
};

pub use cubek_test_utils::{
    HostData, ValidationResult, compare_host_data_files, read_host_data, write_host_data,
};

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
