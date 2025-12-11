#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum LineMode {
    Parallel,
    Perpendicular,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
/// How bound checks is handled for inner reductions.
pub enum BoundChecks {
    /// No bound check is necessary.
    None,
    /// Using a mask is enough for bound checks.
    /// This will still read the memory in an out-of-bound location,
    /// but will replace the value by the null value.
    Mask,
    /// Branching is necessary for bound checks.
    ///
    /// Probably the right setting when performing fuse on read.
    Branch,
}
