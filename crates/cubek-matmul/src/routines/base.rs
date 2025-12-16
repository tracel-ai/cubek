use crate::components::batch::{BatchConfig, BatchMatmulFamily};
use crate::definition::{MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError};
use cubecl::prelude::*;
use std::fmt::Debug;

/// Specifications for a matmul algorithm
pub trait Routine {
    type Strategy: Default + Debug + Clone;
    type Blueprint: Debug + Clone;
    type Config: BatchConfig;

    type BatchMatmul: BatchMatmulFamily<Blueprint = Self::Blueprint, Config = Self::Config>;

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<<Self::BatchMatmul as BatchMatmulFamily>::Config, MatmulSetupError> {
        Self::BatchMatmul::setup(client, problem, selection, line_sizes, dtypes)
    }

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<Self::Blueprint, MatmulSetupError>;

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_max
    }

    // Ideally put this elsewhere
    fn can_cast_stage_element() -> bool;
}
