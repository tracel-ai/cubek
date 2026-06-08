#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AccessPatternKind {
    ReduceAxisPattern(ReduceAxisPatternArgs),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ReduceAxisPatternArgs {
    pub reduce_size: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum GlobalOperationKind {
    Scalar(Semiring),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Semiring {
    Sum,
    Prod,
    Max,
    Min,
    Any,
    All,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MemoryReaderKind {
    Global,
}
