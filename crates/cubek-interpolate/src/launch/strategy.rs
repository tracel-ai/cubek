use crate::routines::{BlueprintStrategy, GlobalMemoryRoutine, SharedMemoryRoutine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateStrategy {
    pub routine: RoutineStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoutineStrategy {
    GlobalMemoryStrategy(BlueprintStrategy<GlobalMemoryRoutine>),
    SharedMemoryStrategy(BlueprintStrategy<SharedMemoryRoutine>),
}
