#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum GlobalOperation {
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
