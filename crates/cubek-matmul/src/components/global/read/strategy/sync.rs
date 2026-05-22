use cubecl::prelude::*;

use crate::{
    components::global::{SharedGlobalMatmulConfig, read::SyncStrategy},
    definition::MatmulTypes,
};

/// Simple synchronous barrier, using `cube_sync()`
pub struct Synchronous {}

#[cube]
impl SyncStrategy for Synchronous {
    type Barrier = ();

    fn create_barrier() -> Self::Barrier {}

    fn sync<MP: MatmulTypes>(
        _barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig,
    ) {
        sync_cube();
    }
}
