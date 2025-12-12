use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    random::random_uniform,
    reduce::{
        components::instructions::ReduceOperationConfig,
        launch::ReduceStrategy,
        routines::{RoutineStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
    },
};
use std::marker::PhantomData;

#[allow(dead_code)]
struct ReduceBench<R: Runtime, E> {
    shape: Vec<usize>,
    device: R::Device,
    axis: usize,
    client: ComputeClient<R>,
    strategy: ReduceStrategy,
    _e: PhantomData<E>,
}

impl<R: Runtime, E: Float> Benchmark for ReduceBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let input = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(&client, 0., 1., input.as_ref(), elem).unwrap();
        let mut shape_out = self.shape.clone();
        shape_out[self.axis] = 1;
        let out = TensorHandle::empty(&client, shape_out, elem);

        (input, out)
    }

    fn execute(&self, (input, out): Self::Input) -> Result<(), String> {
        cubek::reduce::reduce::<R>(
            &self.client,
            input.as_ref(),
            out.as_ref(),
            self.axis,
            self.strategy.clone(),
            ReduceOperationConfig::Sum,
            cubek::reduce::ReduceDtypes {
                input: E::as_type_native_unchecked(),
                output: E::as_type_native_unchecked(),
                accumulation: f32::as_type_native_unchecked(),
            },
        )
        .map_err(|err| format!("{err}"))?;

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}-{:?}",
            self.axis,
            E::as_type_native_unchecked(),
            self.shape,
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: R::Device) {
    let client = R::client(&device);
    for shape in [
        vec![2, 2, 4099 * 32],
        vec![32, 512, 4096],
        vec![4096, 512, 32],
        vec![512, 512],
    ] {
        for axis in 0..shape.len() {
            for strategy in [
                ReduceStrategy::FullUnit(RoutineStrategy::Strategy(UnitStrategy)),
                ReduceStrategy::FullCube(RoutineStrategy::Strategy(CubeStrategy {
                    use_planes: true,
                })),
                ReduceStrategy::FullCube(RoutineStrategy::Strategy(CubeStrategy {
                    use_planes: false,
                })),
                ReduceStrategy::FullPlane(RoutineStrategy::Strategy(PlaneStrategy {
                    independant: true,
                })),
                ReduceStrategy::FullPlane(RoutineStrategy::Strategy(PlaneStrategy {
                    independant: false,
                })),
            ] {
                let bench = ReduceBench::<R, E> {
                    shape: shape.clone(),
                    axis,
                    client: client.clone(),
                    device: device.clone(),
                    strategy,
                    _e: PhantomData,
                };
                println!("Running: ==== {} ====", bench.name());
                match bench.run(TimingMethod::System) {
                    Ok(val) => {
                        println!("{val}");
                    }
                    Err(err) => println!("Can't run the benchmark: {err}"),
                }
            }
        }
    }
}

fn main() {
    run::<cubecl::TestRuntime, f32>(Default::default());
}
