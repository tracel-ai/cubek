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
pub use cubek_matmul::tiled::eval::split_cubes;
pub use cubek_matmul::tiled::eval::split_k;
pub use cubek_matmul::tiled::eval::tile_quant_stage;
pub use cubek_pool::eval::benchmarks as pool;
pub use cubek_random::eval::benchmarks as random;
pub use cubek_reduce::eval::benchmarks as reduce;
pub use cubek_std::eval::benchmarks::contiguous;
pub use cubek_std::eval::benchmarks::memcpy_async;
pub use cubek_std::eval::benchmarks::unary;

pub use cubek_test_utils::{
    BenchmarkCategory, CatalogEntry, Category, Correctness, HostData, ItemDescriptor, ResourceKind,
    RunSamples, ValidationResult, compare_host_data_files, read_host_data, write_host_data,
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
        &crate::split_cubes::Category,
        &crate::split_k::Category,
        &crate::tile_quant_stage::Category,
        &crate::unary::Category,
    ]
}

/// A comma-separated list of substrings from an environment variable, empty
/// when unset, which selects everything.
fn id_filter(var: &str) -> Vec<String> {
    std::env::var(var)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn selects(filter: &[String], id: &str) -> bool {
    filter.is_empty() || filter.iter().any(|want| id.contains(want.as_str()))
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
///
/// `CUBEK_BENCH_PROBLEMS` and `CUBEK_BENCH_STRATEGIES` take a comma-separated
/// list of substrings and run only the ids that contain one. `gemm` alone is
/// 184 problems against 27 strategies.
pub fn run_category(category: &dyn BenchmarkCategory) {
    use cubecl::benchmark::BenchmarkDurations;

    let samples: usize = std::env::var("CUBEK_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let problem_filter = id_filter("CUBEK_BENCH_PROBLEMS");
    let strategy_filter = id_filter("CUBEK_BENCH_STRATEGIES");

    let all_problems = category.problems();
    let all_strategies = category.strategies();
    let problems: Vec<_> = all_problems
        .iter()
        .filter(|p| selects(&problem_filter, &p.id))
        .collect();
    let strategies: Vec<_> = all_strategies
        .iter()
        .filter(|s| selects(&strategy_filter, &s.id))
        .collect();

    if problems.is_empty() || strategies.is_empty() {
        println!(
            "no rows selected: {} problem(s) and {} strategy(ies) survived the filters",
            problems.len(),
            strategies.len()
        );
        return;
    }
    if !problem_filter.is_empty() || !strategy_filter.is_empty() {
        println!(
            "running {} of {} problem(s) against {} of {} strategy(ies)",
            problems.len(),
            all_problems.len(),
            strategies.len(),
            all_strategies.len()
        );
    }

    let warmed = category.warm_peaks(&problems.iter().map(|p| p.id.clone()).collect::<Vec<_>>());
    if warmed > 0 {
        println!("warmed {warmed} device peak(s) before timing");
    }

    let mut any_over_peak = false;
    let mut any_launch_bound = false;

    for problem in &problems {
        for strategy in &strategies {
            println!("---- {} / {} ----", strategy.label, problem.label);
            match category.run(&strategy.id, &problem.id, samples) {
                Ok(samples) => {
                    if let Some(tflops) = samples.tflops {
                        println!("{tflops:.3} TFLOPS");
                    }
                    // Which ceiling the row ran into, rather than only how fast
                    // it went.
                    if let Some(binding) = &samples.binding {
                        let pct = 100.0 * binding.fraction_of_peak;
                        let over_peak = binding.fraction_of_peak > 1.0;
                        any_over_peak |= over_peak;
                        let suffix = if over_peak { ", over peak" } else { "" };
                        let achieved_gb_s = binding.achieved_per_s / 1e9;
                        match binding.resource {
                            ResourceKind::Launch => {
                                any_launch_bound = true;
                                println!("launch overhead: {pct:.0}% of the run{suffix}");
                            }
                            ResourceKind::Compute => {
                                println!("bound by compute ({pct:.0}% of peak{suffix})");
                            }
                            ResourceKind::Read => {
                                println!(
                                    "{achieved_gb_s:.1} GB/s ({pct:.0}% of read peak{suffix})"
                                );
                            }
                            ResourceKind::Write => {
                                println!(
                                    "{achieved_gb_s:.1} GB/s ({pct:.0}% of write peak{suffix})"
                                );
                            }
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

    if any_launch_bound {
        println!(
            "note: a row reporting \"launch overhead\" spent more of its wall time being dispatched than doing the work it declares. The percentage is the share of the run that dispatch cost, so a small one says the run is not explained by anything the roofline knows about, not that the dispatch was cheap."
        );
    }

    if any_over_peak {
        println!(
            "note: \"over peak\" rows beat the measured ceiling. The memory probes stream cold, so a working set that stays in cache can exceed them."
        );
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
