/// How a routine's cubes reach the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulCubeWrite {
    /// Each cell written once, by the one cube that owns it.
    Replace,
    /// The cubes sharing a box take their turns at a carry in the sum's element, each adding its
    /// part into what the turns before it left, and the last hands the sum on to the output, which
    /// is written once ([`Relay`](cubek_tile::launch::Relay)).
    Relay,
    /// A cube holding a box whole writes it once. Each of the cubes sharing a box parks its part in
    /// a slot of the sum's element, and the last to arrive adds the parts in the order of their
    /// runs and hands the sum on to the output, which is written once
    /// ([`LastArrival`](cubek_tile::launch::LastArrival)).
    LastArrival,
}

impl MatmulCubeWrite {
    /// Whether a sink takes this write. A sink is written once a cell and never read back, which
    /// each write gives the output: every box stored once, by one cube, with its final sum.
    ///
    /// The relay declines all the same. Its cubes wait on each other, which can hang a Metal GPU,
    /// so it has not been run into a sink; it takes one once it has, on a device whose cubes may
    /// wait (CUDA).
    pub fn takes_sink(self) -> bool {
        match self {
            Self::Replace | Self::LastArrival => true,
            Self::Relay => false,
        }
    }

    /// Whether a cube adds its part into what its box already holds: a relay's turn adds into the
    /// carry, so its fragments cannot store their cells as they lie.
    pub fn accumulates(self) -> bool {
        match self {
            Self::Relay => true,
            Self::Replace | Self::LastArrival => false,
        }
    }

    /// Whether other cubes write the box a cube writes: the carry or the slots its parts meet in.
    pub fn shares_box(self) -> bool {
        match self {
            Self::Replace => false,
            Self::Relay | Self::LastArrival => true,
        }
    }

    /// Whether a cube walks its run of `K` in one walk, opening its stages once: a relay's cube
    /// walks it a stage box at a time, opening stages for each.
    pub fn walks_once(self) -> bool {
        match self {
            Self::Replace | Self::LastArrival => true,
            Self::Relay => false,
        }
    }
}
