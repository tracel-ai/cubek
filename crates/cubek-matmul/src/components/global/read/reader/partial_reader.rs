use std::marker::PhantomData;

use super::{StageBuffer, TaskCounter};
use crate::{
    components::{
        global::{
            GlobalReaderConfig, SharedGlobalMatmulConfig,
            memory::GlobalIterator,
            multi_stage::{JobExecutor, JobIterator, LoadMaxRoundPlaneCount},
            read::{LoadingJob, LoadingValidation, PartialLoaderStage, SyncBarrier, SyncStrategy},
        },
        stage::{LoadStageFamily, StageConfig, TilingLayout},
    },
    definition::MatmulTypes,
    launch::RuntimeConfig,
};
use cubecl::prelude::{barrier::Barrier, *};
use cubecl::std::tensor::{View, layout::Coords2d};
use cubek_std::tile::TileKind;

#[cube]
/// A strategy for loading partial stage memory
pub trait PartialLoadingStrategy<RC: RuntimeConfig>:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;
    type SyncStrategy: SyncStrategy;
    type Stage: LoadStageFamily<ReadOnly, TileKind = Self::TileKind>;
    type TileKind: TileKind;

    /// The [LoadingJob] for this strategy.
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>: LoadingJob<EG, NG, ES, NS, Self::TilingLayout, Self::SyncStrategy, Stage = Self::Stage>;

    /// Returns the job with preliminary calculations done.
    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        runtime_config: RC,
        #[comptime] stage_index: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS>;
}

#[cube]
/// A strategy for loading partial stage memory with async barriers. Used for specialized.
pub trait AsyncPartialLoadingStrategy<RC: RuntimeConfig>:
    PartialLoadingStrategy<RC, SyncStrategy: SyncStrategy<Barrier = Shared<Barrier>>>
{
    /// Arrival count for initializing the barrier
    fn arrival_count<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> u32;
    /// Extra synchronization after initializing the barrier, if needed
    fn barrier_post_init();
    /// Arrive at the barrier using the correct completion mechanism, without waiting
    fn arrive<MP: MatmulTypes, S: StageConfig>(
        barrier: &mut Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    );
    /// Whether this unit should participate in the load loop
    fn is_elected<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> bool;
}

#[derive(Clone, CubeType)]
#[allow(clippy::type_complexity)]
/// Loads a stage from stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct PartialStageGlobalReader<
    EG: Numeric,
    NG: Size,
    ES: Numeric,
    NS: Size,
    RC: RuntimeConfig,
    L: PartialLoadingStrategy<RC>,
> {
    global_iter: GlobalIterator<Vector<EG, NG>>,
    runtime_config: RC,
    stage_memory: PartialLoaderStage<RC, L, ES, NS>,
    loading_job: ComptimeOption<(L::Job<EG, NG, ES, NS>, L::Job<EG, NG, ES, NS>)>,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, RC: RuntimeConfig, L: PartialLoadingStrategy<RC>>
    PartialStageGlobalReader<EG, NG, ES, NS, RC, L>
{
    /// Create a new SyncPartialStageGlobalReader
    pub fn new(
        tensor: View<Vector<EG, NG>, Coords2d>,
        runtime_config: RC,
        k_step: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self {
        let stage_memory = L::Stage::create(128usize, config.smem_config);
        let global_iter =
            GlobalIterator::new(tensor, k_step, config.gmem_config.view_direction, false);

        let loading_job = match config.precompute_job {
            true => ComptimeOption::new_Some((
                L::new_job::<EG, NG, ES, NS>(runtime_config.clone(), 0u32, config),
                L::new_job::<EG, NG, ES, NS>(runtime_config.clone(), 1u32, config),
            )),
            false => ComptimeOption::new_None(),
        };

        PartialStageGlobalReader::<EG, NG, ES, NS, RC, L> {
            global_iter,
            runtime_config,
            stage_memory,
            loading_job,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(
        &self,
        #[comptime] stage_buffer: StageBuffer,
    ) -> PartialLoaderStage<RC, L, ES, NS> {
        L::Stage::with_buffer_index(&self.stage_memory, stage_buffer.to_index())
    }

    /// Frees the stage memory for reuse
    pub fn free_stage(self) {
        L::Stage::free(&self.stage_memory);
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(
        &mut self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) {
        #[comptime]
        #[comptime]
        let mut loading_job = match self.loading_job.clone() {
            ComptimeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            ComptimeOption::None => L::new_job::<EG, NG, ES, NS>(
                self.runtime_config.clone(),
                stage_buffer.to_index(),
                config,
            ),
        };

        let len = L::Job::task_count(&loading_job);

        #[unroll]
        for task_id in 0..len {
            L::Job::<EG, NG, ES, NS>::execute_task(
                &mut loading_job,
                task_id,
                &self.global_iter,
                &mut self.stage_memory,
                barrier,
                config,
            );
        }
    }
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, RC: RuntimeConfig, L: PartialLoadingStrategy<RC>>
    JobExecutor<L::SyncStrategy> for PartialStageGlobalReader<EG, NG, ES, NS, RC, L>
{
    type JobIterator = PartialJobIterator<EG, NG, ES, NS, RC, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::JobIterator {
        #[comptime]
        let job = match this.loading_job.clone() {
            ComptimeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            ComptimeOption::None => L::new_job::<EG, NG, ES, NS>(
                this.runtime_config.clone(),
                stage_buffer.to_index(),
                config,
            ),
        };

        let num_tasks = L::Job::task_count(&job);

        PartialJobIterator::<EG, NG, ES, NS, RC, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
            _rc: PhantomData,
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut PartialJobIterator<EG, NG, ES, NS, RC, L>,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_id = job_iterator.current.read().counter.comptime();

        L::Job::<EG, NG, ES, NS>::execute_task(
            &mut job_iterator.job,
            task_id,
            &this.global_iter,
            &mut this.stage_memory,
            barrier,
            config,
        );

        job_iterator.current.store(TaskCounter {
            counter: task_id + 1,
        });
    }

    fn execute_all_remaining_tasks(
        this: &mut Self,
        job_iterator: &mut Self::JobIterator,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_counter = job_iterator.current.read().counter;

        #[unroll]
        for task_id in task_counter..job_iterator.num_tasks {
            L::Job::<EG, NG, ES, NS>::execute_task(
                &mut job_iterator.job,
                task_id,
                &this.global_iter,
                &mut this.stage_memory,
                barrier,
                config,
            );
        }

        job_iterator.current.store(TaskCounter {
            counter: job_iterator.num_tasks,
        });
    }

    fn execute_whole_job(
        this: &mut Self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job_iterator(this, stage_buffer, config),
            barrier,
            config,
        );
    }
}

#[derive(CubeType)]
/// Accomplish the entire job of filling the stage
pub struct PartialJobIterator<
    EG: Numeric,
    NG: Size,
    ES: Numeric,
    NS: Size,
    RC: RuntimeConfig,
    L: PartialLoadingStrategy<RC>,
> {
    job: L::Job<EG, NG, ES, NS>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
    #[cube(comptime)]
    _rc: PhantomData<RC>,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size, RC: RuntimeConfig, L: PartialLoadingStrategy<RC>>
    JobIterator for PartialJobIterator<EG, NG, ES, NS, RC, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
