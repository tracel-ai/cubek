use crate::routines::{BlueprintStrategy, GlobalMemoryRoutine, SharedMemoryRoutine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolateStrategy {
    GlobalMemoryStrategy(BlueprintStrategy<GlobalMemoryRoutine>),
    SharedMemoryStrategy(BlueprintStrategy<SharedMemoryRoutine>),
}
