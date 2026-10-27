/// Where a plane microkernel's steps read their keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubePlaneReads {
    /// The cube's stage: the cube copies each step of keys and values into shared memory once
    /// for all its planes, and stages its query before the walk.
    Stage,
    /// Where they lie: each plane reads its own keys of each step, and the cube stages nothing.
    Global,
}

/// How planes of a plane microkernel holding the same rows may take their keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubePlaneKeys {
    /// Split between them, each taking its block of every step, their rows merged once the walk
    /// ends.
    Split,
    /// Whole: a plane walks every key of its rows, so no two planes hold the same rows.
    Whole,
}

/// How a cube lays its planes over its box: a plane for every head it holds, along every plane of
/// a head's queries, along every plane of the same rows' keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePlaneLayout {
    pub heads: usize,
    pub planes_along_queries: usize,
    pub planes_along_keys: usize,
}

impl CubePlaneLayout {
    /// Planes one cube holds.
    pub fn planes(&self) -> usize {
        self.heads * self.planes_along_queries * self.planes_along_keys
    }

    /// Groups of planes holding the same rows, one for every head along every plane of its
    /// queries.
    pub fn row_groups(&self) -> usize {
        self.heads * self.planes_along_queries
    }
}
