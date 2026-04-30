use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    frontend, future,
    ir::MatrixLayout,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    matmul::{
        definition::MatmulElems,
        launch::{Strategy, launch_ref},
    },
    random::random_uniform,
    std::InputBinding,
};

use crate::{
    gemv::{
        problem::{GemvProblem, ProblemKind, problem_for},
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

    let flops = 2.0 * problem.batches as f64 * problem.out_dim as f64 * problem.k_dim as f64;

    let bench = GemvBench::<R> {
        problem,
        strategy,
        device,
        client,
        dtypes: MatmulElems::from_single_dtype(E::as_type_native_unchecked()),
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations).with_flops(flops))
}

struct GemvBench<R: Runtime> {
    problem: GemvProblem,
    strategy: Strategy,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
    samples: usize,
}

#[derive(Clone)]
struct GemvInputs<R: Runtime> {
    lhs: TensorHandle<R>,
    rhs: TensorHandle<R>,
    out: TensorHandle<R>,
}

fn make_tensor_with_layout<R: Runtime>(
    client: &ComputeClient<R>,
    row_major_shape: [usize; 3],
    layout: MatrixLayout,
    dtype: StorageType,
) -> TensorHandle<R> {
    match layout {
        MatrixLayout::RowMajor => {
            let t = TensorHandle::empty(client, row_major_shape, dtype);
            random_uniform(client, 0., 1., t.clone().binding(), t.dtype).unwrap();
            t
        }
        MatrixLayout::ColMajor => {
            let mut col_major_shape = row_major_shape;
            let rank = col_major_shape.len();
            col_major_shape.swap(rank - 2, rank - 1);
            let mut t = TensorHandle::empty(client, col_major_shape, dtype);
            random_uniform(client, 0., 1., t.clone().binding(), t.dtype).unwrap();
            let len = t.metadata.rank();
            t.metadata.strides_mut().swap(len - 2, len - 1);
            t.metadata.shape_mut().swap(len - 2, len - 1);
            t
        }
        MatrixLayout::Undefined => panic!(),
    }
}

impl<R: Runtime> Benchmark for GemvBench<R> {
    type Input = GemvInputs<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let (lhs_row_major_shape, rhs_row_major_shape, out_shape) = match self.problem.kind {
            ProblemKind::VecMat => (
                [self.problem.batches, 1, self.problem.k_dim],
                [
                    self.problem.batches,
                    self.problem.k_dim,
                    self.problem.out_dim,
                ],
                [self.problem.batches, 1, self.problem.out_dim],
            ),
            ProblemKind::MatVec => (
                [
                    self.problem.batches,
                    self.problem.out_dim,
                    self.problem.k_dim,
                ],
                [self.problem.batches, self.problem.k_dim, 1],
                [self.problem.batches, self.problem.out_dim, 1],
            ),
        };

        let lhs = make_tensor_with_layout(
            &client,
            lhs_row_major_shape,
            self.problem.lhs_layout,
            self.dtypes.lhs_global,
        );
        let rhs = make_tensor_with_layout(
            &client,
            rhs_row_major_shape,
            self.problem.rhs_layout,
            self.dtypes.rhs_global,
        );
        let out = TensorHandle::empty(&client, out_shape, self.dtypes.acc_global);

        GemvInputs { lhs, rhs, out }
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        launch_ref(
            &self.strategy,
            &self.client,
            InputBinding::Normal(inputs.lhs.binding(), self.dtypes.lhs_global),
            InputBinding::Normal(inputs.rhs.binding(), self.dtypes.rhs_global),
            inputs.out.clone().binding(),
            &mut self.dtypes.clone(),
        )
        .map_err(|err| format!("{err}"))
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "{:?}-b:{}-out:{}-k:{}-lhs:{:?}-rhs:{:?}",
            self.problem.kind,
            self.problem.batches,
            self.problem.out_dim,
            self.problem.k_dim,
            self.problem.lhs_layout,
            self.problem.rhs_layout,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
