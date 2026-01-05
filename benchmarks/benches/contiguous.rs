use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    future,
    prelude::*,
    std::tensor::TensorHandle,
};

impl<R: Runtime> Benchmark for IntoContiguousBench<R> {
    type Input = TensorHandle<R>;
    type Output = TensorHandle<R>;

    fn prepare(&self) -> Self::Input {
        fn swap(array: &mut [usize], dims: (usize, usize)) {
            let tmp = array[dims.0];
            array[dims.0] = array[dims.1];
            array[dims.1] = tmp;
        }
        let mut handle = TensorHandle::empty(&self.client, self.shape.clone(), self.dtype);
        for dims in self.dims.iter() {
            swap(&mut handle.shape, *dims);
            swap(&mut handle.strides, *dims);
        }

        handle
    }

    fn execute(&self, input: Self::Input) -> Result<TensorHandle<R>, String> {
        cubecl::std::tensor::into_contiguous_ref(&self.client, &input.as_ref(), self.dtype)
            .map_err(|err| format!("{err:}"))
    }

    fn name(&self) -> String {
        format!(
            "into_contiguous-{:?}-{:?}-{:?}-{:?}",
            self.dtype, self.dims, self.device, self.shape,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
struct IntoContiguousBench<R: Runtime> {
    shape: Vec<usize>,
    device: R::Device,
    dims: Vec<(usize, usize)>,
    client: ComputeClient<R>,
    dtype: StorageType,
}

#[allow(dead_code)]
fn run<R: Runtime>(device: R::Device, dtype: StorageType) {
    for shape in [vec![16, 16, 512, 512]] {
        // for shape in [vec![32, 512, 2048], vec![16, 16, 512, 512]] {
        // for dims in get_combinations(shape.len()) {
        let client = R::client(&device);
        let bench = IntoContiguousBench::<R> {
            shape: shape.clone(),
            dims: vec![(1, 2), (2, 3)],
            client,
            device: device.clone(),
            dtype,
        };
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::Device).unwrap());
    }
    //}
}

fn get_combinations(n: usize) -> impl Iterator<Item = (usize, usize)> {
    // Iterate from 0 up to n
    (0..n).flat_map(move |i| {
        // For each i, iterate from i + 1 up to n
        // This ensures no repeats (i == j) and no duplicates (j, i)
        (i + 1..n).map(move |j| (i, j))
    })
}

fn main() {
    run::<cubecl::TestRuntime>(Default::default(), f32::as_type_native_unchecked());
}
