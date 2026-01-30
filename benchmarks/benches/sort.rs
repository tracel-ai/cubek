//! Benchmarks for cubek-sort.
//!
//! Measures throughput in GB/s for sorting operations with high fidelity.

use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, TimingMethod},
    future,
    prelude::*,
};
use cubek::sort::{SortStrategy, sort_keys};
use std::time::Duration;

const NUM_WARMUP: usize = 5;
const NUM_SAMPLES: usize = 50;

struct SortBench<R: Runtime> {
    num_items: usize,
    client: ComputeClient<R>,
    strategy: SortStrategy,
    seed: u64,
}

impl<R: Runtime> Benchmark for SortBench<R> {
    type Input = (cubecl::server::Handle, cubecl::server::Handle);
    type Output = cubecl::server::Handle;

    fn prepare(&self) -> Self::Input {
        // Generate deterministic pseudo-random u32 values using simple LCG
        let data: Vec<u32> = (0..self.num_items)
            .scan(self.seed, |state, _| {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                Some((*state >> 32) as u32)
            })
            .collect();

        let input = self.client.create_from_slice(u32::as_bytes(&data));
        let output = self
            .client
            .empty(self.num_items * std::mem::size_of::<u32>());

        (input, output)
    }

    fn name(&self) -> String {
        format!(
            "sort-u32-{}-{}",
            format_size(self.num_items),
            R::name(&self.client),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed")
    }

    fn execute(&self, (input, output): Self::Input) -> Result<Self::Output, String> {
        let shape = [self.num_items];
        let strides = [1];

        let input_ref = unsafe {
            TensorHandleRef::from_raw_parts(&input, &strides, &shape, std::mem::size_of::<u32>())
        };
        let output_ref = unsafe {
            TensorHandleRef::from_raw_parts(&output, &strides, &shape, std::mem::size_of::<u32>())
        };

        sort_keys::<R, u32>(
            &self.client,
            input_ref,
            output_ref,
            self.num_items,
            Some(self.strategy.clone()),
        )
        .map_err(|e| format!("Sort failed: {:?}", e))?;

        Ok(output)
    }

    fn num_samples(&self) -> usize {
        NUM_SAMPLES
    }
}

fn format_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}m", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

/// Calculate throughput in GB/s for sorting.
/// For radix sort: we read each element 4 times (4 passes) and write 4 times.
/// Total memory movement = num_items * 4 bytes * 4 passes * 2 (read+write) = 32 bytes per element
fn calculate_throughput(num_items: usize, duration: Duration) -> f64 {
    let bytes_per_element = 4; // u32
    let passes = 4;
    let total_bytes = num_items * bytes_per_element * passes * 2; // read + write per pass
    let duration_sec = duration.as_secs_f64();
    if duration_sec > 0.0 {
        (total_bytes as f64) / duration_sec / 1e9
    } else {
        0.0
    }
}

/// Run warmup iterations to stabilize GPU state (clocks, caches, etc.)
fn warmup<R: Runtime>(bench: &SortBench<R>) {
    for _ in 0..NUM_WARMUP {
        let input = bench.prepare();
        let _ = bench.execute(input);
        bench.sync();
    }
}

/// Calculate standard deviation from durations
fn std_dev(durations: &[Duration], mean: Duration) -> Duration {
    if durations.len() < 2 {
        return Duration::ZERO;
    }
    let mean_nanos = mean.as_nanos() as f64;
    let variance: f64 = durations
        .iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>()
        / (durations.len() - 1) as f64;
    Duration::from_nanos(variance.sqrt() as u64)
}

/// Calculate percentile from sorted durations
fn percentile(sorted_durations: &[Duration], p: f64) -> Duration {
    if sorted_durations.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted_durations.len() - 1) as f64 * p / 100.0).round() as usize;
    sorted_durations[idx.min(sorted_durations.len() - 1)]
}

fn run<R: Runtime>(device: R::Device) {
    let client = R::client(&device);
    let strategy = SortStrategy::default();

    let sizes = [
        64 * 1024,        // 64K
        256 * 1024,       // 256K
        1024 * 1024,      // 1M
        4 * 1024 * 1024,  // 4M
        16 * 1024 * 1024, // 16M
    ];

    println!(
        "Sort Benchmark (warmup={}, samples={})",
        NUM_WARMUP, NUM_SAMPLES
    );
    println!("{}", "=".repeat(80));

    for size in sizes {
        let bench = SortBench::<R> {
            num_items: size,
            client: client.clone(),
            strategy: strategy.clone(),
            seed: 12345,
        };

        let name = bench.name();

        // Warmup
        warmup(&bench);

        match bench.run(TimingMethod::System) {
            Ok(bench_durations) => {
                let durations = &bench_durations.durations;
                let computed = BenchmarkComputations::new(&bench_durations);

                // Calculate statistics
                let mean: Duration = durations.iter().sum::<Duration>() / durations.len() as u32;
                let std = std_dev(durations, mean);

                let mut sorted = durations.clone();
                sorted.sort();
                let p5 = percentile(&sorted, 5.0);
                let p95 = percentile(&sorted, 95.0);

                // Throughput calculations
                let throughput_median = calculate_throughput(size, computed.median);
                let throughput_mean = calculate_throughput(size, mean);
                let throughput_best = calculate_throughput(size, computed.min);
                let throughput_p5 = calculate_throughput(size, p5);

                println!("\n{}", name);
                println!("  Items: {:>12}", size);
                println!(
                    "  Time (ms):  median={:>7.2}  mean={:>7.2}  std={:>6.2}  min={:>7.2}  max={:>7.2}",
                    computed.median.as_secs_f64() * 1000.0,
                    mean.as_secs_f64() * 1000.0,
                    std.as_secs_f64() * 1000.0,
                    computed.min.as_secs_f64() * 1000.0,
                    computed.max.as_secs_f64() * 1000.0,
                );
                println!(
                    "  Throughput: median={:>7.2}  mean={:>7.2}  best={:>7.2}  p5={:>7.2} GB/s",
                    throughput_median, throughput_mean, throughput_best, throughput_p5,
                );
                println!(
                    "  Percentiles (ms): p5={:.2}  p50={:.2}  p95={:.2}",
                    p5.as_secs_f64() * 1000.0,
                    computed.median.as_secs_f64() * 1000.0,
                    p95.as_secs_f64() * 1000.0,
                );
            }
            Err(e) => println!("{}: Failed - {}", name, e),
        }
    }

    println!("\n{}", "=".repeat(80));
}

fn main() {
    run::<cubecl::TestRuntime>(Default::default());
}
