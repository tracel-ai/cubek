use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    calculate_cube_count_elemwise, frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::random::random_uniform;
use std::marker::PhantomData;

#[cube(launch)]
fn execute<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>) {
    if ABSOLUTE_POS < out.len() {
        for i in 0..256u32 {
            if i % 2 == 0 {
                out[ABSOLUTE_POS] -= F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            } else {
                out[ABSOLUTE_POS] += F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            }
        }
    }
}

impl<R: Runtime, E: Float> Benchmark for UnaryBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let lhs = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(&client, 0., 1., lhs.as_ref(), elem).unwrap();
        let rhs = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(&client, 0., 1., rhs.as_ref(), elem).unwrap();
        let out = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(&client, 0., 1., out.as_ref(), elem).unwrap();

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Input) -> Result<(), String> {
        let num_elems: usize = out.shape.iter().product();

        let working_units = num_elems / self.vectorization;
        let cube_dim = CubeDim::new(&self.client, working_units);
        let cube_count = calculate_cube_count_elemwise(&self.client, working_units, cube_dim);

        execute::launch::<E, R>(
            &self.client,
            cube_count,
            cube_dim,
            lhs.as_arg(self.vectorization),
            rhs.as_arg(self.vectorization),
            out.as_arg(self.vectorization),
        )
        .unwrap();
        Ok(())
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);

        format!(
            "unary-{}-{}-{:?}",
            R::name(&client),
            E::as_type_native_unchecked(),
            self.vectorization
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "unary-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}

#[allow(dead_code)]
struct UnaryBench<R: Runtime, E> {
    shape: Vec<usize>,
    vectorization: LineSize,
    device: R::Device,
    client: ComputeClient<R>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: R::Device, vectorization: LineSize) {
    let client = R::client(&device);
    let bench = UnaryBench::<R, E> {
        shape: vec![32, 512, 2048],
        vectorization,
        client,
        device,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::Device).unwrap());
}

fn main() {
    run::<cubecl::TestRuntime, f32>(Default::default(), 4);
}
