use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    frontend,
    frontend::Float,
    future,
    prelude::{barrier::Barrier, *},
    std::tensor::TensorHandle,
};
use cubek::random::random_uniform;

use crate::{
    memcpy_async::{
        problem::{MemcpyAsyncProblem, problem_for},
        strategy::{CopyStrategyEnum, strategy_for},
    },
    registry::RunSamples,
};

#[cube]
trait ComputeTask: Send + Sync + 'static {
    fn compute<E: Float, N: Size>(
        input: &Slice<Vector<E, N>>,
        acc: &mut Array<Vector<E, N>>,
        #[comptime] config: Config,
    );

    fn to_output<E: Float, N: Size>(
        acc: &mut Array<Vector<E, N>>,
        output: &mut SliceMut<Vector<E, N>>,
        #[comptime] config: Config,
    );
}

#[derive(CubeType)]
struct DummyCompute {}
#[cube]
impl ComputeTask for DummyCompute {
    fn compute<E: Float, N: Size>(
        input: &Slice<Vector<E, N>>,
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
        output: &mut SliceMut<Vector<E, N>>,
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
    type Barrier: CubeType + Copy + Clone;

    fn barrier() -> Self::Barrier;

    fn memcpy<E: Float, N: Size>(
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        _barrier: Self::Barrier,
        #[comptime] _config: Config,
    ) {
        for i in 0..source.len() {
            destination[i] = source[i];
        }
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;
        let start = UNIT_POS_Y * sub_length;
        let end = start + sub_length;

        barrier.memcpy_async(
            &source.slice(start as usize, end as usize),
            &mut destination.slice_mut(start as usize, end as usize),
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;
        let start = UNIT_POS_Y * sub_length;
        let end = start + sub_length;

        if UNIT_POS_X == 0 {
            barrier.memcpy_async(
                &source.slice(start as usize, end as usize),
                &mut destination.slice_mut(start as usize, end as usize),
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;
        for i in 0..config.num_planes {
            let start = i * sub_length;
            let end = start + sub_length;

            barrier.memcpy_async(
                &source.slice(start as usize, end as usize),
                &mut destination.slice_mut(start as usize, end as usize),
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / config.num_planes;

        if UNIT_POS < config.num_planes {
            let start = UNIT_POS * sub_length;
            let end = start + sub_length;

            barrier.memcpy_async(
                &source.slice(start as usize, end as usize),
                &mut destination.slice_mut(start as usize, end as usize),
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_loops = source.len() as u32 / num_units;

        for i in 0..num_loops {
            let start = UNIT_POS + i * num_units;
            let end = start + 1;

            barrier.memcpy_async(
                &source.slice(start as usize, end as usize),
                &mut destination.slice_mut(start as usize, end as usize),
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
        source: &Slice<Vector<E, N>>,
        destination: &mut SliceMut<Vector<E, N>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() as u32 / (config.num_planes * config.plane_dim);
        let start = UNIT_POS * sub_length;
        let end = start + sub_length;

        barrier.memcpy_async(
            &source.slice(start as usize, end as usize),
            &mut destination.slice_mut(start as usize, end as usize),
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

        Cpy::memcpy(
            &input.slice(start, end),
            &mut smem.to_slice_mut(),
            barrier,
            config,
        );

        Cpy::wait(barrier);

        Cpt::compute(&smem.to_slice(), &mut acc, config);
    }

    Cpy::wait(barrier);
    Cpt::compute(&smem.to_slice(), &mut acc, config);
    Cpt::to_output(&mut acc, &mut output.to_slice_mut(), config);
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
            Cpy::memcpy(
                &input.slice(start, end),
                &mut smem1.to_slice_mut(),
                barrier1,
                config,
            );
            if i > 0 {
                Cpy::wait(barrier2);
                Cpt::compute(&smem2.to_slice(), &mut acc, config);
            }
        } else {
            Cpy::memcpy(
                &input.slice(start, end),
                &mut smem2.to_slice_mut(),
                barrier2,
                config,
            );

            Cpy::wait(barrier1);
            Cpt::compute(&smem1.to_slice(), &mut acc, config);
        }
    }

    Cpy::wait(barrier2);
    Cpt::compute(&smem2.to_slice(), &mut acc, config);
    Cpt::to_output(&mut acc, &mut output.to_slice_mut(), config);
}

fn launch_ref<R: Runtime, E: Float>(
    strategy: CopyStrategyEnum,
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
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
                memcpy_test::launch_unchecked::<E, DummyCopy, DummyCompute, R>(
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
                memcpy_test::launch_unchecked::<E, CoalescedCopy, DummyCompute, R>(
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
                    R,
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
            CopyStrategyEnum::MemcpyAsyncSingleSliceElected => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSingleSliceElected, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSingleSliceElectedCooperative => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSingleSliceElectedCooperative,
                    DummyCompute,
                    R,
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
                    R,
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
            CopyStrategyEnum::MemcpyAsyncSplitPlaneElectedUnit => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSplitPlaneElectedUnit, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitDuplicatedAll => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSplitDuplicatedAll, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitLargeUnitWithIdle => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSplitLargeUnitWithIdle, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    1,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitSmallUnitCoalescedLoop => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitSmallUnitCoalescedLoop,
                    DummyCompute,
                    R,
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
                    R,
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

    let bench = MemcpyAsyncBench::<R, E> {
        problem,
        strategy,
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

struct MemcpyAsyncBench<R: Runtime, E> {
    problem: MemcpyAsyncProblem,
    strategy: CopyStrategyEnum,
    device: R::Device,
    client: ComputeClient<R>,
    samples: usize,
    _e: PhantomData<E>,
}

impl<R: Runtime, E: Float> Benchmark for MemcpyAsyncBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let a = TensorHandle::empty(
            &client,
            vec![self.problem.data_count],
            E::as_type_native_unchecked(),
        );
        random_uniform(
            &client,
            0.,
            1.,
            a.clone().binding(),
            E::as_type_native_unchecked().storage_type(),
        )
        .unwrap();
        let b = TensorHandle::empty(
            &client,
            vec![self.problem.window_size],
            E::as_type_native_unchecked(),
        );
        random_uniform(
            &client,
            0.,
            1.,
            b.clone().binding(),
            E::as_type_native_unchecked().storage_type(),
        )
        .unwrap();

        (a, b)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let smem_size = args.1.shape()[0];
        launch_ref::<R, E>(
            self.strategy,
            &self.client,
            args.0.binding(),
            args.1.binding(),
            smem_size,
            self.problem.double_buffering,
        );
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "memcpy_async-{}-{}-{:?}",
            R::name(&client),
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
