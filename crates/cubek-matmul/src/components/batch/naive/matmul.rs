use crate::{
    components::batch::{BatchConfig, BatchMatmul, CubeCountInput},
    definition::*,
    launch::MatmulArgs,
};
use cubecl::cube;

pub struct NaiveMatmul {}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NaiveMatmulConfig {}

impl BatchConfig for NaiveMatmulConfig {
    type GlobalConfig = ();

    fn global_config(&self) -> Self::GlobalConfig {
        ()
    }

    fn cube_dim(&self) -> cubecl::CubeDim {
        todo!()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        todo!()
    }

    fn hypercube_config(&self) -> crate::components::batch::HypercubeConfig {
        todo!()
    }

    fn can_yield_extra_cubes(&self) -> bool {
        todo!()
    }
}

#[cube]
impl<MP: MatmulPrecision> BatchMatmul<MP> for NaiveMatmul {
    type Config = NaiveMatmulConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
}
