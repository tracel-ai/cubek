//! Benchmarks for cubek-sort.
//!
//! Measures throughput in GB/s for sorting operations.

use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, TimingMethod},
    future,
    prelude::*,
};
use cubek::sort::{SortStrategy, sort_keys};
use std::time::Duration;

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
        10
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

    for size in sizes {
        let bench = SortBench::<R> {
            num_items: size,
            client: client.clone(),
            strategy: strategy.clone(),
            seed: 12345,
        };

        let name = bench.name();
        println!("Running: ==== {} ====", name);

        match bench.run(TimingMethod::System) {
            Ok(durations) => {
                let computed = BenchmarkComputations::new(&durations);
                let throughput_median = calculate_throughput(size, computed.median);
                let throughput_peak = calculate_throughput(size, computed.min);

                println!(
                    "  Items: {} | Median: {:.2} ms | Throughput: {:.2} GB/s (median), {:.2} GB/s (peak)",
                    size,
                    computed.median.as_secs_f64() * 1000.0,
                    throughput_median,
                    throughput_peak
                );
            }
            Err(e) => println!("  Failed: {}", e),
        }
    }
}

fn main() {
    run::<cubecl::TestRuntime>(Default::default());
}
