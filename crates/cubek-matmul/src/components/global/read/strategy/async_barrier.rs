use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::{
    components::{
        global::{SharedGlobalMatmulConfig, read::SyncStrategy},
        stage::StageConfig,
    },
    definition::MatmulPrecision,
};

/// Asynchronous barrier for `async_memcpy`
pub struct AsyncBarrier {}

#[cube]
impl SyncStrategy for AsyncBarrier {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0)
    }

    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.arrive_and_wait();
    }
}

/// Asynchronous barrier for `async_copy`
pub struct AsyncCopy {}

#[cube]
impl SyncStrategy for AsyncCopy {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0)
    }

    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.commit_copy_async();
        barrier.arrive_and_wait();
    }
}
