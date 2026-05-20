#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateStrategy {
    memory_strategy: MemoryStrategy,
}

impl InterpolateStrategy {
    pub fn new(memory_strategy: MemoryStrategy) -> Self {
        Self { memory_strategy }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryStrategy {
    Global,
    Shared,
}
