use std::sync::Arc;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    bytes::Bytes,
    client::ComputeClient,
    features::TypeUsage,
    future,
    ir::ElemType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, TestInput};

use super::problem::DequantizeProblem;
use super::strategy::DequantizePath;
use crate::scheme::{QuantScheme, QuantStore};

const SEED: u64 = 0x1;

pub fn bench(
    strategy: &DequantizePath,
    problem: &DequantizeProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let scheme = problem.scheme;

    if *strategy == DequantizePath::Legacy && scheme.num_levels() > 1 {
        return Err("the legacy kernels serve one scale level".into());
    }
    if scheme.store == QuantStore::Native
        && !i8::supported_uses(&client).contains(TypeUsage::Conversion)
    {
        return Err("the backend has no native i8 conversion".into());
    }
    if *strategy == DequantizePath::Tile && matches!(scheme.store, QuantStore::PackedU32(_)) {
        let widest = client
            .io_optimized_vector_sizes(size_of::<u32>())
            .next()
            .unwrap_or(1);
        if scheme.num_quants() > widest {
            return Err(format!(
                "device lines cap at {widest}, below the packing factor {}",
                scheme.num_quants()
            ));
        }
    }

    let bench = DequantizeBench {
        m: problem.m,
        n: problem.n,
        scheme,
        path: *strategy,
        client,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct DequantizeBench {
    m: usize,
    n: usize,
    scheme: QuantScheme,
    path: DequantizePath,
    client: ComputeClient<TestRuntime>,
    samples: usize,
}

/// One problem's device tensors. The quantized buffer is declared twice because the two
/// entries disagree on units: the legacy kernels take a packed store in `u32` words, the
/// tile entry the same buffer counted in the values it holds (for a native store the two
/// declarations coincide).
struct Inputs {
    stored: TensorHandle<TestRuntime>,
    values: TensorHandle<TestRuntime>,
    /// One tensor per scale level, innermost first.
    scales: Vec<TensorHandle<TestRuntime>>,
    output: TensorHandle<TestRuntime>,
}

impl Benchmark for DequantizeBench {
    type Input = Arc<Inputs>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let shape = vec![self.m, self.n];
        let (stored, values) = match self.scheme.store {
            QuantStore::PackedU32(_) => {
                let pack = self.scheme.num_quants();
                let words: Vec<u32> = (0..self.m * self.n / pack)
                    .map(|i| (i as u32).wrapping_mul(0x9E3779B9))
                    .collect();
                let handle = self.client.create(Bytes::from_elems(words));
                let stored = TensorHandle::new_contiguous(
                    vec![self.m, self.n / pack],
                    handle.clone(),
                    u32::elem_type_native(),
                );
                let values =
                    TensorHandle::new_contiguous(shape.clone(), handle, u32::elem_type_native());
                (stored, values)
            }
            _ => {
                let range = self.scheme.value.range();
                let input = TestInput::builder(self.client.clone(), Shape::from(shape.clone()))
                    .dtype(ElemType::from_quant_value(self.scheme.value))
                    .uniform(SEED, range.0, range.1)
                    .generate_without_host_data();
                (input.clone(), input)
            }
        };

        // One scale tensor per level, innermost first: the block level grids over the shape, the
        // per-tensor level holds a single scale.
        let scales = self
            .scheme
            .block_scale()
            .map(|block| block.size.grid(&[self.m, self.n]))
            .into_iter()
            .chain(self.scheme.tensor_scale().map(|_| vec![1, 1]))
            .map(|grid| {
                let count = grid.iter().product();
                let scale_values: Vec<f32> =
                    (0..count).map(|k| 0.05 * ((k % 64) + 1) as f32).collect();
                TestInput::builder(self.client.clone(), Shape::from(grid))
                    .custom(scale_values)
                    .generate_without_host_data()
            })
            .collect();

        let output = TensorHandle::zeros(&self.client, Shape::from(shape), f32::elem_type_native());
        Arc::new(Inputs {
            stored,
            values,
            scales,
            output,
        })
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let scales: Vec<_> = args.scales.iter().map(|s| s.clone().binding()).collect();
        let output_dtype = f32::elem_type_native();
        match self.path {
            DequantizePath::Legacy => crate::dequantize::launch_legacy(
                &self.client,
                args.stored.clone().binding(),
                args.output.clone().binding(),
                &scales,
                &self.scheme,
                output_dtype,
            ),
            DequantizePath::Tile => crate::dequantize_tiled::launch_ref(
                &self.client,
                args.values.clone().binding(),
                args.output.clone().binding(),
                &scales,
                &self.scheme,
                output_dtype,
            ),
        }
        .map_err(|e| format!("{e:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "dequantize-{}-{:?}-m{}-n{}",
            <TestRuntime as Runtime>::name(&self.client),
            self.path,
            self.m,
            self.n,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "dequantize-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
