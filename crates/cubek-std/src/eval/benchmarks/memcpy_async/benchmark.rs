use std::marker::PhantomData;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    frontend::Float,
    future,
    prelude::{barrier::Barrier, *},
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_test_utils::{RunSamples, StridedLayout, TestInput};

use crate::eval::benchmarks::memcpy_async::problem::MemcpyAsyncProblem;
use crate::eval::benchmarks::memcpy_async::strategy::CopyStrategyEnum;

#[cube]
trait ComputeTask: Send + Sync + 'static {
    fn compute<E: Float, N: Size>(
        input: &[Vector<E, N>],
        acc: &mut Array<Vector<E, N>>,
        #[comptime] config: Config,
    );

    fn to_output<E: Float, N: Size>(
        acc: &mut Array<Vector<E, N>>,
        output: &mut [Vector<E, N>],
        #[comptime] config: Config,
    );
}

#[derive(CubeType)]
struct DummyCompute {}
#[cube]
impl ComputeTask for DummyCompute {
    fn compute<E: Float, N: Size>(
        input: &[Vector<E, N>],
        acc: &mut Array<Vector<E, N>>,
        #[comptime] config: Config,
    ) {
        let offset = 256;
        let position = (UNIT_POS as usize * config.acc_len + offset) % config.smem_size;
        for i in 0..config.acc_len {
            acc[i] += input[position + i];
        }
    }

    fn to_output<E: Float, N: Size>(
        acc: &mut Array<Vector<E, N>>,
        output: &mut [Vector<E, N>],
        #[comptime] config: Config,
    ) {
        let position = UNIT_POS as usize * config.acc_len;
        for i in 0..config.acc_len {
            acc[i] += output[position + i];
        }
    }
}

#[cube]
trait CopyStrategy: Send + Sync + 'static {
    type Barrier: CubeType<ExpandType: Copy> + Copy + Clone;

    fn barrier() -> Self::Barrier;

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    );

    fn wait(_barrier: Self::Barrier);
}

#[derive(CubeType)]
struct DummyCopy {}
#[cube]
impl CopyStrategy for DummyCopy {
    type Barrier = ();

    fn barrier() -> Self::Barrier {}

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        _barrier: Self::Barrier,
        #[comptime] _config: Config,
    ) {
        destination.copy_from_slice(source);
    }

    fn wait(_barrier: Self::Barrier) {
        sync_cube();
    }
}

#[derive(CubeType)]
struct CoalescedCopy {}
#[cube]
impl CopyStrategy for CoalescedCopy {
    type Barrier = ();

    fn barrier() -> Self::Barrier {}

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        _barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_copies_per_unit = source.len() as u32 / num_units;
        for i in 0..num_copies_per_unit {
            let pos = UNIT_POS + i * num_units;
            destination[pos as usize] = source[pos as usize];
        }
    }

    fn wait(_barrier: Self::Barrier) {
        sync_cube();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSingleSliceDuplicatedAll {}
#[cube]
impl CopyStrategy for MemcpyAsyncSingleSliceDuplicatedAll {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] _config: Config,
    ) {
        barrier.memcpy_async(source, destination)
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSingleSliceElected {}
#[cube]
impl CopyStrategy for MemcpyAsyncSingleSliceElected {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] _config: Config,
    ) {
        if UNIT_POS == 0 {
            barrier.memcpy_async(source, destination)
        }
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSingleSliceElectedCooperative {}
#[cube]
impl CopyStrategy for MemcpyAsyncSingleSliceElectedCooperative {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] _config: Config,
    ) {
        if UNIT_POS == 0 {
            barrier.memcpy_async(source, destination)
        }
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSplitPlaneDuplicatedUnit {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitPlaneDuplicatedUnit {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;
        let start = UNIT_POS_Y * sub_length;
        let end = start + sub_length;

        barrier.memcpy_async(
            &source[start as usize..end as usize],
            &mut destination[start as usize..end as usize],
        )
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSplitPlaneElectedUnit {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitPlaneElectedUnit {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;
        let start = UNIT_POS_Y * sub_length;
        let end = start + sub_length;

        if UNIT_POS_X == 0 {
            barrier.memcpy_async(
                &source[start as usize..end as usize],
                &mut destination[start as usize..end as usize],
            )
        }
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSplitDuplicatedAll {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitDuplicatedAll {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;
        for i in 0..config.num_planes {
            let start = i * sub_length;
            let end = start + sub_length;

            barrier.memcpy_async(
                &source[start as usize..end as usize],
                &mut destination[start as usize..end as usize],
            )
        }
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSplitLargeUnitWithIdle {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitLargeUnitWithIdle {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;

        if UNIT_POS < config.num_planes {
            let start = UNIT_POS * sub_length;
            let end = start + sub_length;

            barrier.memcpy_async(
                &source[start as usize..end as usize],
                &mut destination[start as usize..end as usize],
            )
        }
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSplitSmallUnitCoalescedLoop {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitSmallUnitCoalescedLoop {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_loops = source.len() as u32 / num_units;

        for i in 0..num_loops {
            let start = UNIT_POS + i * num_units;
            let end = start + 1;

            barrier.memcpy_async(
                &source[start as usize..end as usize],
                &mut destination[start as usize..end as usize],
            )
        }
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncSplitMediumUnitCoalescedOnce {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitMediumUnitCoalescedOnce {
    type Barrier = Shared<Barrier>;

    fn barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0u32)
    }

    fn memcpy<E: Float, N: Size>(
        source: &[Vector<E, N>],
        destination: &mut [Vector<E, N>],
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / (config.num_planes * config.plane_dim);
        let start = UNIT_POS * sub_length;
        let end = start + sub_length;

        barrier.memcpy_async(
            &source[start as usize..end as usize],
            &mut destination[start as usize..end as usize],
        )
    }

    fn wait(barrier: Self::Barrier) {
        barrier.arrive_and_wait();
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Config {
    plane_dim: u32,
    num_planes: u32,
    smem_size: usize,
    acc_len: usize,
    double_buffering: bool,
}

#[cube(launch_unchecked)]
fn memcpy_test<E: Float, N: Size, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<Vector<E, N>>,
    output: &mut Tensor<Vector<E, N>>,
    #[comptime] config: Config,
) {
    if config.double_buffering {
        memcpy_test_single_buffer::<E, N, Cpy, Cpt>(input, output, config);
    } else {
        memcpy_test_double_buffer::<E, N, Cpy, Cpt>(input, output, config);
    }
}

#[cube]
fn memcpy_test_single_buffer<E: Float, N: Size, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<Vector<E, N>>,
    output: &mut Tensor<Vector<E, N>>,
    #[comptime] config: Config,
) {
    let data_count = input.shape(0);
    let mut acc = Array::<Vector<E, N>>::new(config.acc_len);
    let num_iterations = data_count.div_ceil(config.smem_size);

    let mut smem = SharedMemory::<Vector<E, N>>::new(config.smem_size);
    let barrier = Cpy::barrier();

    for i in 0..num_iterations {
        let start = i * config.smem_size;
        let end = start + config.smem_size;

        Cpy::memcpy(&input[start..end], smem.as_mut_slice(), barrier, config);

        Cpy::wait(barrier);

        Cpt::compute(smem.as_slice(), &mut acc, config);
    }

    Cpy::wait(barrier);
    Cpt::compute(smem.as_slice(), &mut acc, config);
    Cpt::to_output(&mut acc, output.as_mut_slice(), config);
}

#[cube]
fn memcpy_test_double_buffer<E: Float, N: Size, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<Vector<E, N>>,
    output: &mut Tensor<Vector<E, N>>,
    #[comptime] config: Config,
) {
    let data_count = input.shape(0);
    let mut smem1 = SharedMemory::<Vector<E, N>>::new(config.smem_size);
    let mut smem2 = SharedMemory::<Vector<E, N>>::new(config.smem_size);
    let mut acc = Array::<Vector<E, N>>::new(config.acc_len);
    let num_iterations = data_count.div_ceil(config.smem_size);

    let barrier1 = Cpy::barrier();
    let barrier2 = Cpy::barrier();

    for i in 0..num_iterations {
        let start = i * config.smem_size;
        let end = if start + config.smem_size < data_count {
            start + config.smem_size
        } else {
            data_count
        };

        if i % 2 == 0 {
            Cpy::memcpy(&input[start..end], smem1.as_mut_slice(), barrier1, config);
            if i > 0 {
                Cpy::wait(barrier2);
                Cpt::compute(smem2.as_slice(), &mut acc, config);
            }
        } else {
            Cpy::memcpy(&input[start..end], smem2.as_mut_slice(), barrier2, config);

            Cpy::wait(barrier1);
            Cpt::compute(smem1.as_slice(), &mut acc, config);
        }
    }

    Cpy::wait(barrier2);
    Cpt::compute(smem2.as_slice(), &mut acc, config);
    Cpt::to_output(&mut acc, output.as_mut_slice(), config);
}

fn launch_ref<E: Float>(
    strategy: CopyStrategyEnum,
    client: &ComputeClient<TestRuntime>,
    input: TensorBinding<TestRuntime>,
    output: TensorBinding<TestRuntime>,
    smem_size: usize,
    double_buffering: bool,
) {
    let cube_count = CubeCount::Static(1, 1, 1);
    let plane_dim = 32;
    let num_planes = 8;
    let cube_dim = CubeDim::new_2d(plane_dim, num_planes);
    let config = Config {
        plane_dim,
        num_planes,
        smem_size,
        acc_len: smem_size / (plane_dim * num_planes) as usize,
        double_buffering,
    };

    unsafe {
        match strategy {
            CopyStrategyEnum::DummyCopy => {
                memcpy_test::launch_unchecked::<E, DummyCopy, DummyCompute, TestRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::CoalescedCopy => {
                memcpy_test::launch_unchecked::<E, CoalescedCopy, DummyCompute, TestRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSingleSliceDuplicatedAll => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSingleSliceDuplicatedAll,
                    DummyCompute,
                    TestRuntime,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSingleSliceElected => memcpy_test::launch_unchecked::<
                E,
                MemcpyAsyncSingleSliceElected,
                DummyCompute,
                TestRuntime,
            >(
                client,
                cube_count,
                cube_dim,
                1,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                config,
            ),
            CopyStrategyEnum::MemcpyAsyncSingleSliceElectedCooperative => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSingleSliceElectedCooperative,
                    DummyCompute,
                    TestRuntime,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitPlaneDuplicatedUnit => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitPlaneDuplicatedUnit,
                    DummyCompute,
                    TestRuntime,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitPlaneElectedUnit => memcpy_test::launch_unchecked::<
                E,
                MemcpyAsyncSplitPlaneElectedUnit,
                DummyCompute,
                TestRuntime,
            >(
                client,
                cube_count,
                cube_dim,
                1,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                config,
            ),
            CopyStrategyEnum::MemcpyAsyncSplitDuplicatedAll => memcpy_test::launch_unchecked::<
                E,
                MemcpyAsyncSplitDuplicatedAll,
                DummyCompute,
                TestRuntime,
            >(
                client,
                cube_count,
                cube_dim,
                1,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                config,
            ),
            CopyStrategyEnum::MemcpyAsyncSplitLargeUnitWithIdle => memcpy_test::launch_unchecked::<
                E,
                MemcpyAsyncSplitLargeUnitWithIdle,
                DummyCompute,
                TestRuntime,
            >(
                client,
                cube_count,
                cube_dim,
                1,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                config,
            ),
            CopyStrategyEnum::MemcpyAsyncSplitSmallUnitCoalescedLoop => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitSmallUnitCoalescedLoop,
                    DummyCompute,
                    TestRuntime,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitMediumUnitCoalescedOnce => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitMediumUnitCoalescedOnce,
                    DummyCompute,
                    TestRuntime,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
        };
    }
}

pub fn bench(
    strategy: &CopyStrategyEnum,
    problem: &MemcpyAsyncProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let bench = MemcpyAsyncBench::<f32> {
        data_count: problem.data_count,
        window_size: problem.window_size,
        double_buffering: problem.double_buffering,
        strategy: *strategy,
        client,
        device,
        samples: num_samples,
        _e: PhantomData,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

struct MemcpyAsyncBench<E> {
    data_count: usize,
    window_size: usize,
    double_buffering: bool,
    strategy: CopyStrategyEnum,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    _e: PhantomData<E>,
}

fn make_uniform_1d<E: Float>(
    client: &ComputeClient<TestRuntime>,
    len: usize,
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), Shape::from(vec![len]))
        .layout(StridedLayout::Explicit(vec![1]))
        .dtype(E::as_type_native_unchecked().storage_type())
        .uniform(seed, 0., 1.)
        .generate_without_host_data()
}

impl<E: Float> Benchmark for MemcpyAsyncBench<E> {
    type Input = (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = <TestRuntime as Runtime>::client(&self.device);

        let a = make_uniform_1d::<E>(&client, self.data_count, 0);
        let b = make_uniform_1d::<E>(&client, self.window_size, 1);

        (a, b)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let smem_size = args.1.shape()[0];
        launch_ref::<E>(
            self.strategy,
            &self.client,
            args.0.binding(),
            args.1.binding(),
            smem_size,
            self.double_buffering,
        );
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = <TestRuntime as Runtime>::client(&self.device);
        format!(
            "memcpy_async-{}-{}-{:?}",
            <TestRuntime as Runtime>::name(&client),
            E::as_type_native_unchecked(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "memcpy-async-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
