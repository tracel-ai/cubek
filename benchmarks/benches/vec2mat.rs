use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    matmul::{
        definition::MatmulElems,
        launch::{MatmulInputBinding, launch_vec2mat},
    },
    random::random_uniform,
};

#[allow(dead_code)]
struct Vec2MatBench<R: Runtime> {
    batches: usize,
    n: usize,
    k: usize,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
}

#[derive(Clone)]
struct Vec2MatInputs<R: Runtime> {
    lhs: TensorHandle<R>,
    rhs: TensorHandle<R>,
    out: TensorHandle<R>,
}

impl<R: Runtime> Benchmark for Vec2MatBench<R> {
    type Input = Vec2MatInputs<R>;
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

        Vec2MatInputs { lhs, rhs, out }
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        launch_vec2mat::launch_ref(
            &self.client,
            MatmulInputBinding::Normal(inputs.lhs.binding(), self.dtypes.lhs_global),
            MatmulInputBinding::Normal(inputs.rhs.binding(), self.dtypes.rhs_global),
            inputs.out.clone().binding(),
            &self.dtypes,
        )
        .map_err(|err| format!("{err}"))
    }

    fn name(&self) -> String {
        format!("vec2mat-b:{}-n:{}-k:{}", self.batches, self.n, self.k,).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: R::Device) {
    let client = R::client(&device);

    let bench = Vec2MatBench::<R> {
        client: client.clone(),
        batches: 2,
        n: 4096,
        k: 8192,
        device: device.clone(),
        dtypes: MatmulElems::from_single_dtype(E::as_type_native_unchecked()),
    };
    match bench.run(TimingMethod::System) {
        Ok(val) => {
            println!("{val}");
        }
        Err(err) => println!("Can't run the benchmark: {err}"),
    }
}

fn main() {
    run::<cubecl::TestRuntime, f32>(Default::default());
}
