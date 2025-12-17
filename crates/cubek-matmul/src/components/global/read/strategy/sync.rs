use cubecl::prelude::*;

use crate::{
    components::{
        global::{SharedGlobalMatmulConfig, read::SyncStrategy},
        stage::StageConfig,
    },
    definition::MatmulPrecision,
};

/// Simple synchronous barrier, using `cube_sync()`
pub struct Synchronous {}

#[cube]
impl SyncStrategy for Synchronous {
    type Barrier = ();

    fn create_barrier() -> Self::Barrier {}

    fn sync<MP: MatmulPrecision, S: StageConfig>(
        _barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        sync_cube();
    }
}
