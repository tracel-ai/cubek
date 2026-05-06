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
    gemv::problem::{GemvProblem, ProblemKind},
    registry::RunSamples,
};

pub fn bench(
    strategy: &Strategy,
    problem: &GemvProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    bench_on::<cubecl::TestRuntime, f32>(Default::default(), strategy, problem, num_samples)
}

pub fn bench_on<R: Runtime, E: frontend::Float>(
    device: R::Device,
    strategy: &Strategy,
    problem: &GemvProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let client = R::client(&device);

    let flops = 2.0 * problem.batches as f64 * problem.out_dim as f64 * problem.k_dim as f64;

    let bench = GemvBench::<R> {
        kind: problem.kind,
        batches: problem.batches,
        out_dim: problem.out_dim,
        k_dim: problem.k_dim,
        lhs_layout: problem.lhs_layout,
        rhs_layout: problem.rhs_layout,
        strategy: strategy.clone(),
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
    kind: ProblemKind,
    batches: usize,
    out_dim: usize,
    k_dim: usize,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
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

        let (lhs_row_major_shape, rhs_row_major_shape, out_shape) = match self.kind {
            ProblemKind::VecMat => (
                [self.batches, 1, self.k_dim],
                [self.batches, self.k_dim, self.out_dim],
                [self.batches, 1, self.out_dim],
            ),
            ProblemKind::MatVec => (
                [self.batches, self.out_dim, self.k_dim],
                [self.batches, self.k_dim, 1],
                [self.batches, self.out_dim, 1],
            ),
        };

        let lhs = make_tensor_with_layout(
            &client,
            lhs_row_major_shape,
            self.lhs_layout,
            self.dtypes.lhs_global,
        );
        let rhs = make_tensor_with_layout(
            &client,
            rhs_row_major_shape,
            self.rhs_layout,
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
            self.kind, self.batches, self.out_dim, self.k_dim, self.lhs_layout, self.rhs_layout,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}
