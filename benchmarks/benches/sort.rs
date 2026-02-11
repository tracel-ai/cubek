//! Benchmarks for cubek-sort.

use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, TimingMethod},
    future,
    prelude::*,
    server::Handle,
};
use cubek::sort::{SortKey, SortOrder, SortValues, sort};
use std::marker::PhantomData;
use std::time::Duration;

const NUM_SAMPLES: usize = 100;

struct SortBench<R: Runtime, K> {
    num_items: usize,
    client: ComputeClient<R>,
    _key: PhantomData<K>,
}

impl<R: Runtime, K> Benchmark for SortBench<R, K>
where
    K: SortKey + CubeElement + Numeric,
    K::Radix: SortKey<Radix = K::Radix>,
{
    type Input = Handle;
    type Output = Handle;

    fn prepare(&self) -> Self::Input {
        // Sequential reversed data, wrapping for smaller types
        let elem_bytes = std::mem::size_of::<K>();
        let max_val = if elem_bytes >= 8 {
            u64::MAX
        } else {
            (1u64 << (elem_bytes * 8)) - 1
        };
        let data: Vec<K> = (0..self.num_items as u64)
            .rev()
            .map(|i| K::from_int((i % (max_val + 1)) as i64))
            .collect();
        self.client.create_from_slice(K::as_bytes(&data))
    }

    fn name(&self) -> String {
        format!(
            "sort-{}-{}-{}",
            std::any::type_name::<K>().split("::").last().unwrap_or("?"),
            format_size(self.num_items),
            R::name(&self.client),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed")
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let shape = [self.num_items];
        let strides = [1];

        let input_ref = unsafe {
            TensorHandleRef::from_raw_parts(&input, &strides, &shape, std::mem::size_of::<K>())
        };

        let output = sort::<R, K>(
            &self.client,
            input_ref,
            SortValues::None,
            SortOrder::Ascending,
        )
        .map_err(|e| format!("Sort failed: {:?}", e))?;

        Ok(output.keys)
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

fn run<R: Runtime, K>(client: &ComputeClient<R>, sizes: &[usize])
where
    K: SortKey + CubeElement + Numeric,
    K::Radix: SortKey<Radix = K::Radix>,
{
    let key_bits = std::mem::size_of::<K>() * 8;
    println!("--- {key_bits}-bit keys ---");

    for &size in sizes {
        let bench = SortBench::<R, K> {
            num_items: size,
            client: client.clone(),
            _key: PhantomData,
        };

        match bench.run(TimingMethod::System) {
            Ok(bench_durations) => {
                let durations = &bench_durations.durations;
                let computed = BenchmarkComputations::new(&bench_durations);

                let total_time: Duration = durations.iter().sum();
                let mean: Duration = total_time / durations.len() as u32;

                let duration_sec = total_time.as_secs_f64();
                let keys_per_sec = (size as f64) / duration_sec * (NUM_SAMPLES as f64);

                println!(
                    "{:>4}M: {:.2E} keys/sec  (mean={:.2}ms  min={:.2}ms  max={:.2}ms)",
                    size / 1_000_000,
                    keys_per_sec,
                    mean.as_secs_f64() * 1000.0,
                    computed.min.as_secs_f64() * 1000.0,
                    computed.max.as_secs_f64() * 1000.0,
                );
            }
            Err(e) => {
                println!("{:>4}M: Failed - {}", size / 1_000_000, e);
            }
        }
    }
    println!();
}

fn main() {
    use cubecl::ir::{ElemType, UIntKind};

    let client = cubecl::TestRuntime::client(&Default::default());

    let sizes: Vec<usize> = vec![
        1 << 24, // 16M
        1 << 25, // 32M
        1 << 26, // 64M
    ];

    run::<cubecl::TestRuntime, u32>(&client, &sizes);
    if client
        .properties()
        .features
        .supports_type(ElemType::UInt(UIntKind::U16))
    {
        run::<cubecl::TestRuntime, u16>(&client, &sizes);
    }
}
