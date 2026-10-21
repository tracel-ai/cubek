/// A cube's planes, as a kernel laying out shared memory per plane is compiled against them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePlanes {
    pub count: usize,
    /// Units in one plane.
    pub width: usize,
}
