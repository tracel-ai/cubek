/// How a routine's cubes reach the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulCubeWrite {
    /// Each cell written once, by the one cube that owns it.
    Replace,
    /// A cube holding a box whole writes it once. Each of the cubes sharing a box parks its part in
    /// a slot of the sum's element, and the last to arrive adds the parts in the order of their
    /// runs and hands the sum on to the output, which is written once
    /// ([`LastArrival`](cubecl::tile::launch::LastArrival)).
    LastArrival,
}

impl MatmulCubeWrite {
    /// Whether other cubes write the box a cube writes: the slots its parts meet in.
    pub fn shares_box(self) -> bool {
        match self {
            Self::Replace => false,
            Self::LastArrival => true,
        }
    }
}
