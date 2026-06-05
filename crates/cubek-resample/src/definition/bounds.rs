#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Return a fixed identity value (0 for Linear, -inf for Max, +inf for Min).
    Constant,

    /// Clamp to the nearest valid edge index.
    Replicate,

    /// Reflect the index back into bounds.
    Reflect,

    /// Wrap the index around (periodic).
    Circular,
}
