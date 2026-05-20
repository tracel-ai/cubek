use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateStrategy {
    memory_strategy: MemoryStrategy,
}

impl InterpolateStrategy {
    pub fn new(memory_strategy: MemoryStrategy) -> Self {
        Self { memory_strategy }
    }

    pub fn memory_strategy(&self) -> MemoryStrategy {
        self.memory_strategy
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum MemoryStrategy {
    Global,
    Shared,
}
