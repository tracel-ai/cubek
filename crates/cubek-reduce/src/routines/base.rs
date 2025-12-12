use crate::{LineMode, ReduceDtypes, ReduceError, routines::ReduceBlueprint};
use cubecl::prelude::*;

#[derive(Debug)]
pub struct ReduceLineSettings {
    pub line_mode: LineMode,
    pub line_size_input: u8,
    pub line_size_output: u8,
}

#[derive(Debug)]
pub struct ReduceLaunchSettings {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub line: ReduceLineSettings,
}

#[derive(Debug)]
pub struct ReduceProblem {
    pub vector_size: u32,
    pub vector_count: u32,
    pub axis: u32,
    pub dtypes: ReduceDtypes,
}

#[derive(Debug, Clone)]
pub enum RoutineStrategy<R: Routine> {
    Forced(R::Blueprint, CubeDim),
    Strategy(R::Strategy),
}

pub trait Routine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: RoutineStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError>;
}
