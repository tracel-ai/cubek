//! The encoding of a plane-resident tile ([`PlaneForm`]).

use crate::{MmaIOConfig, RegisterBlock};

/// The encoding of a plane-resident tile: the software instruction's register array (whose
/// execution `config` rides along), or a matrix fragment in one of the two hardware forms. `io`
/// rides the manual form because it comes from a device query, which cannot run in-kernel.
///
/// Never stated by a kernel: each accumulator or fragment constructor names its form
/// ([`Tile::cmma_accumulator`](crate::Tile::cmma_accumulator),
/// [`PlanePartition::mma_fragments`](crate::PlanePartition::mma_fragments), ...), and this is
/// what they hand the encoding-blind partition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum PlaneForm {
    Registers { config: RegisterBlock },
    Cmma,
    Mma { io: MmaIOConfig },
}
