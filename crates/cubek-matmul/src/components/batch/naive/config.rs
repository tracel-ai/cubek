use cubecl::CubeDim;

use crate::{
    components::{
        batch::BatchConfig,
        global::{GlobalReaderConfig, GlobalWriterConfig},
    },
    definition::{HypercubeBlueprint, MatmulLineSizes},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NaiveMatmulConfig {}

impl BatchConfig for NaiveMatmulConfig {
    fn cube_dim(&self) -> CubeDim {
        todo!()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        todo!()
    }

    fn hypercube_blueprint(&self) -> HypercubeBlueprint {
        todo!()
    }

    fn can_yield_extra_cubes(&self) -> bool {
        todo!()
    }

    fn lhs_global_reader_config(&self) -> GlobalReaderConfig {
        todo!()
    }

    fn rhs_global_reader_config(&self) -> GlobalReaderConfig {
        todo!()
    }

    fn global_writer_config(&self) -> GlobalWriterConfig {
        todo!()
    }
}
