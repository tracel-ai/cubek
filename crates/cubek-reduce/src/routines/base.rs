use crate::{ReduceDtypes, ReduceError, VectorizationMode, routines::ReduceBlueprint};
use cubecl::prelude::*;

#[derive(Debug)]
pub struct ReduceVectorSettings {
    pub vectorization_mode: VectorizationMode,
    pub vector_size_input: VectorSize,
    pub vector_size_output: VectorSize,
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
    /// The address type, defined by the max of each handle's `required_address_type`
    pub address_type: AddressType,
}

#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: Routine> {
    Forced(R::Blueprint, CubeDim),
    Inferred(R::Strategy),
}

pub trait Routine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError>;
}
