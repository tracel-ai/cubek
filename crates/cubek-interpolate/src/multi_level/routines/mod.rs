mod blueprint;
mod global_memory;
mod shared_memory;

use crate::{
    InterpolateError,
    definition::InterpolateForwardProblem,
    multi_level::settings::InterpolateLaunchSettings,
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlueprintStrategy<R: ForwardRoutine> {
    Forced(R::Blueprint),
    Inferred(R::Strategy),
}

pub trait ForwardRoutine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        strategy: BlueprintStrategy<Self>,
        vector_size: usize,
        bytes_per_element: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError>;
}

pub use blueprint::*;
pub use global_memory::*;
pub use shared_memory::*;
