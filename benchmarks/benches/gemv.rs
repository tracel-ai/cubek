use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    frontend, future,
    ir::MatrixLayout,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    matmul::{
        definition::MatmulElems,
        launch::{Strategy, launch_ref},
        routines::{
            BlueprintStrategy, TileSizeSelection, simple::SimpleArgs,
            simple_unit::SimpleUnitSelectionArgs, vecmat_plane_parallel::GemvPlaneParallelStrategy,
            vecmat_unit_perpendicular::GemvUnitPerpendicularStrategy,
        },
    },
    random::random_uniform,
    std::InputBinding,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProblemKind {
    VecMat, // [b, 1, k] x [b, k, n] -> [b, 1, n]
    MatVec, // [b, m, k] x [b, k, 1] -> [b, m, 1]
}

#[allow(dead_code)]
struct GemvBench<R: Runtime> {
    batches: usize,
    out_dim: usize,
    k_dim: usize,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
    strategy: Strategy,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    kind: ProblemKind,
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

    fn name(&self) -> String {
        format!(
            "{:?}-b:{}-out:{}-k:{}-lhs:{:?}-rhs:{:?}",
            self.kind, self.batches, self.out_dim, self.k_dim, self.lhs_layout, self.rhs_layout
        )
        .to_lowercase()
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: &R::Device, strategy: Strategy) {
    let client = R::client(device);

    for kind in [ProblemKind::VecMat, ProblemKind::MatVec] {
        println!("{:?}:", kind);
        for layout in [MatrixLayout::RowMajor, MatrixLayout::ColMajor] {
            let (lhs_layout, rhs_layout) = match kind {
                ProblemKind::VecMat => (MatrixLayout::RowMajor, layout), // matrix is rhs
                ProblemKind::MatVec => (layout, MatrixLayout::RowMajor), // matrix is lhs
            };
            println!("  matrix layout={:?}:", layout);

            let bench = GemvBench::<R> {
                client: client.clone(),
                batches: 2,
                out_dim: 4096,
                k_dim: 8192,
                device: device.clone(),
                dtypes: MatmulElems::from_single_dtype(E::as_type_native_unchecked()),
                strategy: strategy.clone(),
                lhs_layout,
                rhs_layout,
                kind,
            };
            match bench.run(TimingMethod::System) {
                Ok(val) => println!("{val}"),
                Err(err) => println!("Can't run the benchmark: {err}"),
            }
        }
    }
}

#[allow(unused)]
fn run_algos_gemv<R: Runtime, E: frontend::Float>(device: &R::Device) {
    println!("Gemv Unit Perpendicular");
    run::<R, E>(
        device,
        Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    );

    println!("===================\n");
    println!("Gemv Plane Parallel");
    run::<R, E>(
        device,
        Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(GemvPlaneParallelStrategy {
            target_num_planes: 8,
        })),
    );

    // println!("===================\n");
    // println!("Simple VecMat");
    // run::<R, E>(
    //     device,
    //     Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
    // );

    // println!("===================\n");
    // println!("Double VecMat");
    // run::<R, E>(
    //     device,
    //     Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
    // );

    // println!("===================\n");
    // println!("Simple Unit Min");
    // run::<R, E>(
    //     device,
    //     Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
    //         tile_size: TileSizeSelection::MinTileSize,
    //     })),
    // );

    // println!("===================\n");
    // println!("Simple Unit Max");
    // run::<R, E>(
    //     device,
    //     Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
    //         tile_size: TileSizeSelection::MaxTileSize,
    //     })),
    // );

    // println!("===================\n");
    // println!("CMMA");
    // run::<R, E>(
    //     device,
    //     Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
    //         multi_rows: false,
    //     })),
    // );
}

fn main() {
    run_algos_gemv::<cubecl::TestRuntime, f32>(&Default::default());
}
