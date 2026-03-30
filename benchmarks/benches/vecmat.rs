use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    matmul::{
        definition::MatmulElems,
        launch::{Strategy, launch_ref},
        routines::{
            BlueprintStrategy, TileSizeSelection, nostage_vecmat::NoStageVecMatStrategy,
            simple_unit::SimpleUnitSelectionArgs,
        },
    },
    random::random_uniform,
    std::InputBinding,
};

#[allow(dead_code)]
struct VecMatBench<R: Runtime> {
    batches: usize,
    n: usize,
    k: usize,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
    strategy: Strategy,
}

#[derive(Clone)]
struct VecMatInputs<R: Runtime> {
    lhs: TensorHandle<R>,
    rhs: TensorHandle<R>,
    out: TensorHandle<R>,
}

impl<R: Runtime> Benchmark for VecMatBench<R> {
    type Input = VecMatInputs<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let lhs = TensorHandle::empty(&client, [self.batches, 1, self.k], self.dtypes.lhs_global);
        random_uniform(&client, 0., 1., lhs.clone().binding(), lhs.dtype).unwrap();

        let rhs = TensorHandle::empty(
            &client,
            [self.batches, self.k, self.n],
            self.dtypes.rhs_global,
        );
        random_uniform(&client, 0., 1., rhs.clone().binding(), rhs.dtype).unwrap();

        let out = TensorHandle::empty(&client, [self.batches, 1, self.n], self.dtypes.acc_global);

        VecMatInputs { lhs, rhs, out }
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
        format!("vecmat-b:{}-n:{}-k:{}", self.batches, self.n, self.k,).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: &R::Device, strategy: Strategy) {
    let client = R::client(device);

    let bench = VecMatBench::<R> {
        client: client.clone(),
        batches: 2,
        n: 4096,
        k: 8192,
        device: device.clone(),
        dtypes: MatmulElems::from_single_dtype(E::as_type_native_unchecked()),
        strategy,
    };
    match bench.run(TimingMethod::System) {
        Ok(val) => {
            println!("{val}");
        }
        Err(err) => println!("Can't run the benchmark: {err}"),
    }
}

#[allow(unused)]
fn run_algos_vecmat<R: Runtime, E: frontend::Float>(device: &R::Device) {
    println!("No Stage VecMat");
    run::<R, E>(
        device,
        Strategy::NoStageVecMat(BlueprintStrategy::Inferred(NoStageVecMatStrategy {
            target_num_planes: 8,
        })),
    );

    println!("Simple VecMat");
    run::<R, E>(
        device,
        Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
    );

    println!("Double VecMat");
    run::<R, E>(
        device,
        Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
    );

    println!("Simple Unit Min");
    run::<R, E>(
        device,
        Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
    );

    println!("Simple Unit Max");
    run::<R, E>(
        device,
        Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
    );
}

fn main() {
    run_algos_vecmat::<cubecl::TestRuntime, f32>(&Default::default());
}
