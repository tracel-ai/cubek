use cubecl::{
    prelude::barrier::{Barrier, BarrierToken},
    prelude::*,
};

use crate::components::global::{GlobalConfig, SharedGlobalMatmulConfig, read::SyncStrategy};
use crate::definition::{LhsS, MatmulTypes, RhsS};

/// Asynchronous barrier for TMA loads
pub struct AsyncTma {}

#[cube]
impl SyncStrategy for AsyncTma {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        let bar = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        bar
    }

    fn sync<MP: MatmulTypes>(
        barrier: &mut Self::Barrier,
        #[comptime] config: SharedGlobalMatmulConfig,
    ) {
        let lhs_elem_size = LhsS::<MP>::type_size().comptime();
        let rhs_elem_size = RhsS::<MP>::type_size().comptime();
        let lhs_bytes =
            config.lhs_reader_config().smem_config.elements_per_stage() * lhs_elem_size as u32;
        let rhs_bytes =
            config.rhs_reader_config().smem_config.elements_per_stage() * rhs_elem_size as u32;
        let num_bytes = lhs_bytes + rhs_bytes;
        let token = arrive_tma(barrier.inner_ref(), num_bytes);
        barrier.wait(token);
    }
}

#[cube]
/// Barrier for TMA
pub fn arrive_tma(barrier: &Barrier, #[comptime] num_bytes: u32) -> BarrierToken {
    let expected = select(UNIT_POS == 0, num_bytes, 0);
    barrier.arrive_and_expect_tx(1, expected)
}
