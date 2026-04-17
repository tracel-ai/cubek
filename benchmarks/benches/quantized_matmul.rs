use std::time::Duration;

use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, TimingMethod},
    frontend, future,
    ir::StorageType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        launch::{Strategy, launch_ref as matmul_launch_ref},
        routines::{
            BlueprintStrategy, TileSizeSelection, double_buffering::DoubleBufferingArgs,
            ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
            simple_unit::SimpleUnitSelectionArgs, vecmat_plane_parallel::GemvPlaneParallelStrategy,
            vecmat_unit_perpendicular::GemvUnitPerpendicularStrategy,
        },
    },
    quantization::{
        quantize,
        scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
    },
    random::random_uniform,
    std::InputBinding,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantSide {
    LhsOnly,
    RhsOnly,
    Both,
}

#[derive(Clone, Copy)]
enum Mode {
    Float,
    Quant {
        scheme: QuantScheme,
        side: QuantSide,
    },
}

#[derive(Clone)]
struct BenchSpec {
    name: &'static str,
    mode: Mode,
}

fn scheme_tensor(value: QuantValue) -> QuantScheme {
    QuantScheme::default()
        .with_mode(QuantMode::Symmetric)
        .with_level(QuantLevel::Tensor)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
}

fn scheme_block(value: QuantValue, block: u8) -> QuantScheme {
    scheme_tensor(value).with_level(QuantLevel::block([block]))
}

#[allow(dead_code)]
struct QuantMatmulBench<R: Runtime> {
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    spec: BenchSpec,
    strategy: Strategy,
    strategy_label: String,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
}

struct QuantOperand<R: Runtime> {
    data: TensorHandle<R>,
    scale: TensorHandle<R>,
    shape: Shape,
    scheme: QuantScheme,
}

enum Operand<R: Runtime> {
    Float(TensorHandle<R>),
    Quant(QuantOperand<R>),
}

impl<R: Runtime> Clone for Operand<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Float(t) => Self::Float(t.clone()),
            Self::Quant(q) => Self::Quant(QuantOperand {
                data: q.data.clone(),
                scale: q.scale.clone(),
                shape: q.shape.clone(),
                scheme: q.scheme,
            }),
        }
    }
}

impl<R: Runtime> Operand<R> {
    fn into_binding(self) -> InputBinding<R> {
        match self {
            Operand::Float(t) => InputBinding::Normal(t.clone().binding(), t.dtype),
            Operand::Quant(q) => InputBinding::Quantized {
                data: q.data.clone().binding(),
                data_dtype: q.data.dtype,
                scale: q.scale.clone().binding(),
                scale_dtype: q.scale.dtype,
                shape: q.shape,
                scheme: q.scheme,
            },
        }
    }
}

#[derive(Clone)]
struct QuantMatmulInputs<R: Runtime> {
    lhs: Operand<R>,
    rhs: Operand<R>,
    out: TensorHandle<R>,
}

fn scales_shape(scheme: &QuantScheme, shape: &[usize]) -> Vec<usize> {
    match &scheme.level {
        QuantLevel::Tensor => vec![1; shape.len()],
        QuantLevel::Block(block) => {
            let rank = shape.len();
            let block_dims = block.to_dim_vec(rank);
            shape
                .iter()
                .zip(block_dims.iter())
                .map(|(d, b)| d / (*b as usize))
                .collect()
        }
    }
}

fn quantize_operand<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorHandle<R>,
    scheme: &QuantScheme,
) -> QuantOperand<R> {
    let shape: Shape = input.shape().clone();
    let scale_shape_vec = scales_shape(scheme, &shape);

    let f32_dtype = f32::as_type_native_unchecked().storage_type();
    let scale_in = TensorHandle::empty(client, scale_shape_vec.clone(), f32_dtype);
    let (q_min, q_max) = scheme.value.range();
    let max_abs_q = q_max.abs().max(q_min.abs()) as f32;
    let base = 1.0 / max_abs_q;
    // Use a narrow band around the nominal tensor-wise scale so block scales are
    // non-degenerate without producing NaNs on quantization.
    random_uniform(
        client,
        base * 0.8,
        base * 1.2,
        scale_in.clone().binding(),
        scale_in.dtype,
    )
    .unwrap();

    let scale_out = TensorHandle::empty(client, scale_shape_vec, f32_dtype);

    let output_dtype = match &scheme.store {
        QuantStore::PackedU32(_) => u32::as_type_native_unchecked().storage_type(),
        other => panic!("benchmark only exercises PackedU32, got {other:?}"),
    };

    let mut quant_shape: Vec<usize> = shape.to_vec();
    let num_quants = scheme.num_quants();
    if num_quants > 1 {
        let last = quant_shape.len() - 1;
        quant_shape[last] /= num_quants;
    }
    let data = TensorHandle::empty(client, quant_shape, output_dtype);

    let input_elem = match input.dtype {
        StorageType::Scalar(e) => e,
        other => panic!("unexpected input storage type {other:?}"),
    };

    quantize::launch_ref(
        client,
        input.binding(),
        data.clone().binding(),
        scale_in.binding(),
        scale_out.clone().binding(),
        scheme,
        input_elem,
    )
    .expect("quantize launch failed");

    QuantOperand {
        data,
        scale: scale_out,
        shape,
        scheme: *scheme,
    }
}

fn float_operand<R: Runtime>(
    client: &ComputeClient<R>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<R> {
    let t = TensorHandle::empty(client, shape, dtype);
    random_uniform(client, -1.0, 1.0, t.clone().binding(), t.dtype).unwrap();
    t
}

impl<R: Runtime> Benchmark for QuantMatmulBench<R> {
    type Input = QuantMatmulInputs<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = &self.client;
        let lhs_shape = vec![self.b, self.m, self.k];
        let rhs_shape = vec![self.b, self.k, self.n];
        let out_shape = vec![self.b, self.m, self.n];

        let lhs_float = float_operand(client, lhs_shape, self.dtypes.lhs_global);
        let rhs_float = float_operand(client, rhs_shape, self.dtypes.rhs_global);

        let (lhs, rhs) = match self.spec.mode {
            Mode::Float => (Operand::Float(lhs_float), Operand::Float(rhs_float)),
            Mode::Quant { scheme, side } => {
                let lhs = match side {
                    QuantSide::LhsOnly | QuantSide::Both => {
                        Operand::Quant(quantize_operand(client, lhs_float, &scheme))
                    }
                    QuantSide::RhsOnly => Operand::Float(lhs_float),
                };
                let rhs = match side {
                    QuantSide::RhsOnly | QuantSide::Both => {
                        Operand::Quant(quantize_operand(client, rhs_float, &scheme))
                    }
                    QuantSide::LhsOnly => Operand::Float(rhs_float),
                };
                (lhs, rhs)
            }
        };

        let out = TensorHandle::empty(client, out_shape, self.dtypes.acc_global);

        QuantMatmulInputs { lhs, rhs, out }
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        matmul_launch_ref(
            &self.strategy,
            &self.client,
            inputs.lhs.into_binding(),
            inputs.rhs.into_binding(),
            inputs.out.clone().binding(),
            &mut self.dtypes.clone(),
        )
        .map_err(|err| format!("{err:?}"))
    }

    fn name(&self) -> String {
        let scheme_part = match self.spec.mode {
            Mode::Float => "float".to_string(),
            Mode::Quant { side, .. } => format!("{}-{:?}", self.spec.name, side),
        };
        format!(
            "quant-matmul-{}-{}-b:{}-m:{}-n:{}-k:{}",
            scheme_part, self.strategy_label, self.b, self.m, self.n, self.k
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<cubecl::benchmark::ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "quant-matmul-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

fn matmul_elems<E: frontend::Float>() -> MatmulElems {
    // Quantized inputs carry their own packed dtype through `InputBinding::Quantized`;
    // `MatmulElems` must describe the float computation dtype end-to-end. `from_globals`
    // also preserves an f32 accumulator when E is f16/bf16, which `from_single_dtype`
    // would incorrectly clobber.
    let dtype = E::as_type_native_unchecked().storage_type();
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    })
}

fn run_one<R: Runtime, E: frontend::Float>(
    client: &ComputeClient<R>,
    device: &R::Device,
    spec: BenchSpec,
    strategy: Strategy,
    strategy_label: &str,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Duration, String> {
    let bench = QuantMatmulBench::<R> {
        b,
        m,
        n,
        k,
        spec,
        strategy,
        strategy_label: strategy_label.to_string(),
        device: device.clone(),
        client: client.clone(),
        dtypes: matmul_elems::<E>(),
    };

    bench
        .run(TimingMethod::System)
        .map(|durations| BenchmarkComputations::new(&durations).median)
}

fn row_label(spec: &BenchSpec, strategy_label: &str) -> String {
    let scheme_part = match spec.mode {
        Mode::Float => "float".to_string(),
        Mode::Quant { side, .. } => format!("{}-{:?}", spec.name, side).to_lowercase(),
    };
    format!("{scheme_part} / {strategy_label}")
}

fn print_table(rows: &[(String, Result<Duration, String>)]) {
    let label_width = rows
        .iter()
        .map(|(l, _)| l.len())
        .max()
        .unwrap_or(0)
        .max("algo / strategy".len());
    println!(
        "{:<label_width$}  {}",
        "algo / strategy",
        "median",
        label_width = label_width
    );
    println!(
        "{:-<label_width$}  {:-<12}",
        "",
        "",
        label_width = label_width
    );
    for (label, result) in rows {
        match result {
            Ok(d) => println!(
                "{:<label_width$}  {:?}",
                label,
                d,
                label_width = label_width
            ),
            Err(_) => println!("{:<label_width$}  ERROR", label, label_width = label_width),
        }
    }
}

fn quant_schemes() -> Vec<(&'static str, QuantScheme)> {
    vec![
        ("q8s-tensor", scheme_tensor(QuantValue::Q8S)),
        ("q4s-tensor", scheme_tensor(QuantValue::Q4S)),
        ("q8s-block32", scheme_block(QuantValue::Q8S, 32)),
        ("q4s-block32", scheme_block(QuantValue::Q4S, 32)),
    ]
}

fn all_specs() -> Vec<BenchSpec> {
    let mut specs = vec![BenchSpec {
        name: "float",
        mode: Mode::Float,
    }];
    for (name, scheme) in quant_schemes() {
        for side in [QuantSide::LhsOnly, QuantSide::RhsOnly, QuantSide::Both] {
            specs.push(BenchSpec {
                name,
                mode: Mode::Quant { scheme, side },
            });
        }
    }
    specs
}

fn gemm_strategies() -> Vec<(&'static str, Strategy)> {
    vec![
        (
            "simple-cyclic-cmma",
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
            })),
        ),
        (
            "ordered-double-cmma",
            Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
            })),
        ),
    ]
}

fn gemv_strategies() -> Vec<(&'static str, Strategy)> {
    vec![
        (
            "gemv-plane-parallel",
            Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(GemvPlaneParallelStrategy {
                target_num_planes: None,
            })),
        ),
        (
            "simple-cyclic-cmma",
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
            })),
        ),
    ]
}

fn run_suite<R: Runtime, E: frontend::Float>(
    client: &ComputeClient<R>,
    device: &R::Device,
    label: &str,
    shapes: &[(usize, usize, usize, usize)],
    strategies: &[(&'static str, Strategy)],
) {
    println!("===== {label} =====");
    let specs = all_specs();
    for &(b, m, n, k) in shapes {
        println!();
        println!("--- shape b={b} m={m} n={n} k={k} ---");
        let mut rows: Vec<(String, Result<Duration, String>)> = Vec::new();
        for (strategy_label, strategy) in strategies {
            for spec in &specs {
                let label = row_label(spec, strategy_label);
                println!("  running {label}");
                let result = run_one::<R, E>(
                    client,
                    device,
                    spec.clone(),
                    strategy.clone(),
                    strategy_label,
                    b,
                    m,
                    n,
                    k,
                );
                rows.push((label, result));
            }
        }
        println!();
        print_table(&rows);
    }
}

fn run_benches<R: Runtime, E: frontend::Float>(device: &R::Device) {
    let client = R::client(device);

    // Inner dims divisible by 32 to accommodate block32 schemes.
    let gemm_shapes: &[(usize, usize, usize, usize)] = &[
        (1, 1024, 1024, 1024),
        (1, 4096, 4096, 4096),
        (2, 1024, 1024, 1024),
    ];

    let gemv_shapes: &[(usize, usize, usize, usize)] =
        &[(1, 1, 4096, 4096), (1, 1, 4096, 8192), (1, 1, 8192, 8192)];

    run_suite::<R, E>(&client, device, "GEMM", gemm_shapes, &gemm_strategies());
    run_suite::<R, E>(&client, device, "GEMV", gemv_shapes, &gemv_strategies());
}

fn main() {
    let device = Default::default();
    println!("########## f32 ##########");
    run_benches::<cubecl::TestRuntime, f32>(&device);
    println!();
    println!("########## f16 ##########");
    run_benches::<cubecl::TestRuntime, half::f16>(&device);
}
