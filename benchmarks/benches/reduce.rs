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
        launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy},
        routines::{
            BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy,
        },
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
    config: ReduceOperationConfig,
    _e: PhantomData<E>,
}

impl<R: Runtime, E: Float> Benchmark for ReduceBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let input = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(
            &client,
            0.,
            1.,
            input.clone().binding(),
            elem.storage_type(),
        )
        .unwrap();
        let mut shape_out = self.shape.clone();
        let reduce_len = match self.config {
            ReduceOperationConfig::ArgTopK(len) => len,
            ReduceOperationConfig::TopK(len) => len,
            _ => 1,
        };
        shape_out[self.axis] = reduce_len;
        let out = TensorHandle::empty(&client, shape_out, elem);

        (input, out)
    }

    fn execute(&self, (input, out): Self::Input) -> Result<(), String> {
        cubek::reduce::reduce::<R>(
            &self.client,
            input.binding(),
            out.binding(),
            self.axis,
            self.strategy.clone(),
            self.config,
            cubek::reduce::ReduceDtypes {
                input: E::as_type_native_unchecked().storage_type(),
                output: E::as_type_native_unchecked().storage_type(),
                accumulation: f32::as_type_native_unchecked().storage_type(),
            },
        )
        .map_err(|err| format!("{err}"))?;

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}-{:?}-{:?}",
            self.axis,
            E::as_type_native_unchecked(),
            self.shape,
            self.strategy,
            self.config,
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
        vec![32, 512, 4095],
        // vec![2, 2, 4099 * 32],
        // vec![4096, 512, 32],
        // vec![512, 512],
    ] {
        for vectorization in [
            VectorizationStrategy {
                parallel_output_vectorization: false,
            },
            VectorizationStrategy {
                parallel_output_vectorization: true,
            },
        ] {
            for strategy in [
                ReduceStrategy {
                    routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                    vectorization,
                },
                ReduceStrategy {
                    routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                        independent: true,
                    })),
                    vectorization,
                },
                // ReduceStrategy {
                //     routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                //         independent: false,
                //     })),
                //     vectorization,
                // },
                ReduceStrategy {
                    routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                        use_planes: true,
                    })),
                    vectorization,
                },
                // ReduceStrategy {
                //     routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                //         use_planes: false,
                //     })),
                //     vectorization,
                // },
            ] {
                for axis in 2..shape.len() {
                    let mut configs = vec![ReduceOperationConfig::Sum; 1];
                    for k in 1..4 {
                        configs.push(ReduceOperationConfig::ArgTopK(k));
                        //configs.push(ReduceOperationConfig::TopK(k));
                    }
                    for config in configs {
                        let bench = ReduceBench::<R, E> {
                            shape: shape.clone(),
                            axis,
                            client: client.clone(),
                            device: device.clone(),
                            strategy: strategy.clone(),
                            config,
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
    }
}

fn main() {
    run::<cubecl::TestRuntime, f32>(Default::default());
}
