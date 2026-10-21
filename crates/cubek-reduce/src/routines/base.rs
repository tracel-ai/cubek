use crate::{
    ReduceDtypes, ReduceError, VectorizationMode, components::instructions::ReduceOperationConfig,
    routines::ReduceBlueprint,
};
use cubecl::prelude::*;

#[derive(Debug)]
pub struct ReduceVectorSettings {
    pub vectorization_mode: VectorizationMode,
    pub vector_size_input: VectorSize,
    pub vector_size_output: VectorSize,
    /// See [`ReduceStrategy::autotune_level`](crate::ReduceStrategy): gates
    /// the unchecked comptime fast paths in the blueprint derivations.
    pub unchecked_fast_paths: bool,
    /// Whether reading the input has side effects (fuse-on-read: the read also
    /// writes a materialized intermediate).
    pub fuse_on_read: bool,
}

#[derive(Debug)]
pub struct ReduceLaunchSettings {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub address_type: AddressType,
    pub vector: ReduceVectorSettings,
}

#[derive(Debug)]
pub struct ReduceProblem {
    /// Number of elements in reduce axis
    pub reduce_len: usize,
    /// Number of instances of the reduce axis
    pub reduce_count: usize,
    pub axis: usize,
    pub dtypes: ReduceDtypes,
    /// The reduce operation being launched. The blueprint selector bounds the cube
    /// width by this instruction's per-accumulator shared-memory footprint (which
    /// scales with `k` for `ArgTopK` / `TopK`).
    pub instruction: ReduceOperationConfig,
    /// The address type, defined by the max of each handle's `required_address_type`
    pub address_type: AddressType,
}

pub use cubek_std::BlueprintStrategy;

/// A reduce routine. A forced blueprint comes with the cube it runs on.
pub trait Routine:
    cubek_std::Routine<
        Strategy: core::fmt::Debug + Clone + Send + 'static,
        Blueprint: core::fmt::Debug + Clone + Send + 'static,
    > + core::fmt::Debug
    + Clone
    + Sized
{
    fn prepare(
        &self,
        client: &Client,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError>;
}
