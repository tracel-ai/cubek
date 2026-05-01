use std::panic::{AssertUnwindSafe, catch_unwind};

use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
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
    },
    quantization::{
        quantize,
        scheme::{QuantLevel, QuantScheme, QuantStore},
    },
    random::random_uniform,
    std::InputBinding,
};

use crate::{
    quantized_matmul::{
        problem::{Layout, Mode, QuantSide, QuantizedMatmulProblem, problem_for},
        strategy::strategy_for,
    },
    registry::RunSamples,
};

pub fn run(strategy_id: &str, problem_id: &str, num_samples: usize) -> Result<RunSamples, String> {
    run_on::<cubecl::TestRuntime, f32>(Default::default(), strategy_id, problem_id, num_samples)
}

pub fn run_on<R: Runtime, E: frontend::Float>(
    device: R::Device,
    strategy_id: &str,
    problem_id: &str,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);
    let problem =
        problem_for(problem_id).ok_or_else(|| format!("unknown problem: {problem_id}"))?;
    let strategy =
        strategy_for(strategy_id).ok_or_else(|| format!("unknown strategy: {strategy_id}"))?;

    validate_spec(&problem)?;

    let flops = 2.0 * problem.b as f64 * problem.m as f64 * problem.n as f64 * problem.k as f64;

    let _ = device;
    let bench = QuantMatmulBench::<R> {
        problem,
        strategy,
        client,
        dtypes: matmul_elems::<E>(),
        samples: num_samples,
    };

    // Some combos still trigger panics inside kernel expansion; catch them so a
    // single bad entry doesn't kill the whole run.
    let durations = match catch_unwind(AssertUnwindSafe(|| bench.run(TimingMethod::System))) {
        Ok(res) => res.map_err(|e| format!("benchmark failed: {e}"))?.durations,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "panic".to_string());
            return Err(format!("panic: {msg}"));
        }
    };

    Ok(RunSamples::new(durations).with_flops(flops))
}

struct QuantMatmulBench<R: Runtime> {
    problem: QuantizedMatmulProblem,
    strategy: Strategy,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
    samples: usize,
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
    lhs_layout: Layout,
    rhs_layout: Layout,
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
    let max_abs_q = q_max.abs().max(q_min.abs());
    let base = 1.0 / max_abs_q;
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

/// For a ColMajor operand we allocate with the last two dims swapped and then
/// swap them back on the `InputBinding`, which uniformly handles shape, strides,
/// scale dims, and the quant packing axis for both float and quantized paths.
fn alloc_shape(logical: &[usize], layout: Layout) -> Vec<usize> {
    let mut s = logical.to_vec();
    if layout == Layout::ColMajor {
        let n = s.len();
        s.swap(n - 2, n - 1);
    }
    s
}

fn to_binding<R: Runtime>(op: Operand<R>, layout: Layout) -> InputBinding<R> {
    let mut binding = op.into_binding();
    if layout == Layout::ColMajor {
        let rank = binding.data().shape.len();
        binding.swap_dims(rank - 2, rank - 1);
    }
    binding
}

impl<R: Runtime> Benchmark for QuantMatmulBench<R> {
    type Input = QuantMatmulInputs<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = &self.client;
        let lhs_logical = vec![self.problem.b, self.problem.m, self.problem.k];
        let rhs_logical = vec![self.problem.b, self.problem.k, self.problem.n];
        let out_shape = vec![self.problem.b, self.problem.m, self.problem.n];

        let lhs_alloc = alloc_shape(&lhs_logical, self.problem.lhs_layout);
        let rhs_alloc = alloc_shape(&rhs_logical, self.problem.rhs_layout);

        let lhs_float = float_operand(client, lhs_alloc, self.dtypes.lhs_global);
        let rhs_float = float_operand(client, rhs_alloc, self.dtypes.rhs_global);

        let (lhs, rhs) = match self.problem.mode {
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

        QuantMatmulInputs {
            lhs,
            rhs,
            lhs_layout: self.problem.lhs_layout,
            rhs_layout: self.problem.rhs_layout,
            out,
        }
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        matmul_launch_ref(
            &self.strategy,
            &self.client,
            to_binding(inputs.lhs, inputs.lhs_layout),
            to_binding(inputs.rhs, inputs.rhs_layout),
            inputs.out.clone().binding(),
            &mut self.dtypes.clone(),
        )
        .map_err(|err| format!("{err:?}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "quant-matmul-{}-{}{}-b:{}-m:{}-n:{}-k:{}",
            self.problem.mode_label.as_str(),
            self.problem.lhs_layout.short(),
            self.problem.rhs_layout.short(),
            self.problem.b,
            self.problem.m,
            self.problem.n,
            self.problem.k,
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
    let dtype = E::as_type_native_unchecked().storage_type();
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    })
}

/// Rejects quant combos whose allocation leaves the packing axis (the last
/// alloc-space dim) too small for `num_quants`, or whose block-size divides
/// a dim unevenly. Both cases otherwise trip divide-by-zero panics deep in
/// the quant/matmul kernels.
fn validate_spec(problem: &QuantizedMatmulProblem) -> Result<(), String> {
    let Mode::Quant { scheme, side } = problem.mode else {
        return Ok(());
    };

    let check = |label: &str, shape: &[usize]| -> Result<(), String> {
        let last = *shape.last().unwrap();
        let nq = scheme.num_quants();
        if last < nq || !last.is_multiple_of(nq) {
            return Err(format!(
                "{label} pack axis={last} incompatible with num_quants={nq}"
            ));
        }
        if let QuantLevel::Block(_) = &scheme.level {
            let scales = scales_shape(&scheme, shape);
            if scales.contains(&0) {
                return Err(format!("{label} block size exceeds a dim in {shape:?}"));
            }
        }
        Ok(())
    };

    if matches!(side, QuantSide::LhsOnly | QuantSide::Both) {
        check(
            "lhs",
            &alloc_shape(&[problem.b, problem.m, problem.k], problem.lhs_layout),
        )?;
    }
    if matches!(side, QuantSide::RhsOnly | QuantSide::Both) {
        check(
            "rhs",
            &alloc_shape(&[problem.b, problem.k, problem.n], problem.rhs_layout),
        )?;
    }
    Ok(())
}
