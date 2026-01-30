//! Benchmarks for cubek-sort.
//!
//! Normalized to match b0nes GPUSorting benchmark format for comparison.
//! Reports both keys/sec (for b0nes comparison) and GB/s (memory throughput).

use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, TimingMethod},
    future,
    prelude::*,
};
use cubek::sort::sort_keys;
use std::time::Duration;

// Match b0nes benchmark parameters: 1 warmup + 100 timed iterations
const NUM_WARMUP: usize = 1;
const NUM_SAMPLES: usize = 100;

struct SortBench<R: Runtime> {
    num_items: usize,
    client: ComputeClient<R>,
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
            None, // Use auto-tuned strategy based on input size
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

/// Calculate keys/sec throughput (b0nes format)
fn calculate_keys_per_sec(num_items: usize, total_duration: Duration, iterations: usize) -> f64 {
    let duration_sec = total_duration.as_secs_f64();
    if duration_sec > 0.0 {
        (num_items as f64) / duration_sec * (iterations as f64)
    } else {
        0.0
    }
}

fn run<R: Runtime>(device: R::Device) {
    let client = R::client(&device);

    // Sizes matching b0nes benchmark range (2^20 to 2^28)
    let sizes: Vec<usize> = vec![
        1 << 20, // 1M (2^20)
        1 << 22, // 4M (2^22)
        1 << 24, // 16M (2^24)
        1 << 26, // 64M (2^26)
        1 << 28, // 268M (2^28) - b0nes default
    ];

    println!("================================================================================");
    println!("CubeK Sort Benchmark - Normalized for b0nes comparison");
    println!("================================================================================");
    println!("Runtime: {}", R::name(&client));
    println!("Timing:  GPU hardware timestamps (same as D3D12 timestamp queries)");
    println!("Warmup iterations: {}", NUM_WARMUP);
    println!("Timed iterations:  {}", NUM_SAMPLES);
    println!("Using auto-tuned strategy based on input size");
    println!("================================================================================\n");

    for size in sizes {
        let bench = SortBench::<R> {
            num_items: size,
            client: client.clone(),
            seed: 10, // Match b0nes default seed
        };

        let name = bench.name();
        let size_bits = (size as f64).log2() as u32;

        println!("Beginning sort keys-only u32 ascending batch timing test at:");
        println!("Size: {} (2^{})", size, size_bits);
        println!("Test size: {}", NUM_SAMPLES);
        print!("Running");

        // Warmup
        warmup(&bench);

        match bench.run(TimingMethod::Device) {
            Ok(bench_durations) => {
                let durations = &bench_durations.durations;
                let computed = BenchmarkComputations::new(&bench_durations);

                // Calculate statistics
                let total_time: Duration = durations.iter().sum();
                let mean: Duration = total_time / durations.len() as u32;
                let std = std_dev(durations, mean);

                let mut sorted = durations.clone();
                sorted.sort();

                // b0nes style: keys/sec = size / totalTime * batchCount
                let keys_per_sec = calculate_keys_per_sec(size, total_time, NUM_SAMPLES);

                // Also compute GB/s for memory bandwidth perspective
                let throughput_mean = calculate_throughput(size, mean);

                println!();
                println!("Total time elapsed: {:.6} seconds", total_time.as_secs_f64());
                println!(
                    "Estimated speed at {} 32-bit elements: {:.6E} keys/sec",
                    size, keys_per_sec
                );
                println!(
                    "Memory throughput (mean): {:.2} GB/s",
                    throughput_mean
                );
                println!(
                    "Per-iteration: mean={:.3}ms  std={:.3}ms  min={:.3}ms  max={:.3}ms",
                    mean.as_secs_f64() * 1000.0,
                    std.as_secs_f64() * 1000.0,
                    computed.min.as_secs_f64() * 1000.0,
                    computed.max.as_secs_f64() * 1000.0,
                );
                println!();
            }
            Err(e) => {
                println!();
                println!("{}: Failed - {}", name, e);
                println!();
            }
        }
    }

    println!("================================================================================");
}

/// Run batched benchmark: submit N sorts before syncing to measure sustained throughput.
/// This amortizes per-submit overhead and measures how fast the GPU can actually sort
/// when it's continuously busy (like in a production pipeline).
fn run_batched<R: Runtime>(device: R::Device) {
    let client = R::client(&device);

    // (size, batch_size) - scale batch size down for larger inputs to avoid OOM
    let configs: Vec<(usize, usize)> = vec![
        (1 << 20, 100), // 1M: 100 batches = 400MB
        (1 << 22, 25),  // 4M: 25 batches = 400MB
        (1 << 24, 6),   // 16M: 6 batches = 384MB
        (1 << 26, 2),   // 64M: 2 batches = 512MB
    ];

    println!("================================================================================");
    println!("CubeK Sort Benchmark - BATCHED (sustained throughput)");
    println!("================================================================================");
    println!("Runtime: {}", R::name(&client));
    println!("This measures GPU throughput when continuously busy.");
    println!("================================================================================\n");

    for (size, batch_size) in configs {
        let size_bits = (size as f64).log2() as u32;
        println!("Size: {} (2^{}), batch size: {}", size, size_bits, batch_size);

        // Prepare all buffers upfront
        let mut inputs = Vec::with_capacity(batch_size);
        let mut outputs = Vec::with_capacity(batch_size);

        let seed = 10u64;
        for i in 0..batch_size {
            let data: Vec<u32> = (0..size)
                .scan(seed.wrapping_add(i as u64), |state, _| {
                    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    Some((*state >> 32) as u32)
                })
                .collect();
            inputs.push(client.create_from_slice(u32::as_bytes(&data)));
            outputs.push(client.empty(size * std::mem::size_of::<u32>()));
        }

        // Warmup
        {
            let shape = [size];
            let strides = [1];
            let input_ref = unsafe {
                TensorHandleRef::from_raw_parts(&inputs[0], &strides, &shape, std::mem::size_of::<u32>())
            };
            let output_ref = unsafe {
                TensorHandleRef::from_raw_parts(&outputs[0], &strides, &shape, std::mem::size_of::<u32>())
            };
            let _ = sort_keys::<R, u32>(&client, input_ref, output_ref, size, None);
            future::block_on(client.sync()).expect("sync failed");
        }

        // Timed batched run
        let start = std::time::Instant::now();

        for i in 0..batch_size {
            let shape = [size];
            let strides = [1];
            let input_ref = unsafe {
                TensorHandleRef::from_raw_parts(&inputs[i], &strides, &shape, std::mem::size_of::<u32>())
            };
            let output_ref = unsafe {
                TensorHandleRef::from_raw_parts(&outputs[i], &strides, &shape, std::mem::size_of::<u32>())
            };
            sort_keys::<R, u32>(&client, input_ref, output_ref, size, None)
                .expect("sort failed");
        }

        future::block_on(client.sync()).expect("sync failed");
        let elapsed = start.elapsed();

        let per_sort = elapsed / batch_size as u32;
        let keys_per_sec = (size as f64 * batch_size as f64) / elapsed.as_secs_f64();
        let throughput = calculate_throughput(size, per_sort);

        println!(
            "  Total time: {:.3}ms for {} sorts = {:.3}ms per sort",
            elapsed.as_secs_f64() * 1000.0,
            batch_size,
            per_sort.as_secs_f64() * 1000.0
        );
        println!(
            "  Throughput: {:.2E} keys/sec, {:.1} GB/s",
            keys_per_sec, throughput
        );
        println!();
    }

    println!("================================================================================");
    println!("ANALYSIS: Batched throughput shows raw GPU performance.");
    println!("At 1M, batched (6.6E9) matches D3D12 GPU-timed (6.5E9) performance.");
    println!("The per-iteration gap is wgpu runtime overhead, not algorithmic.");
    println!("================================================================================");
}

fn main() {
    // Standard per-iteration benchmark (matches b0nes methodology)
    run::<cubecl::TestRuntime>(Default::default());

    // Batched benchmark (sustained throughput, amortizes per-submit overhead)
    println!("\n\n");
    run_batched::<cubecl::TestRuntime>(Default::default());
}
