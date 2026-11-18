//! The dimension labels of a matmul's space, defined once.
//!
//! A [`Space`] is only meaningful against the labels it was built with, so every matmul, its
//! quantized operands and its storage tiling label the same dimensions the same way. Batch
//! dimensions are numbered after the three leading ones ([`batch_dimension`]); the blocked
//! dimensions are numbered past the batch slots, since labels only have to be distinct within the
//! space that uses them.

use cubecl::tile::{Dimension, Space};

/// Rows of the output: the lhs's outer dimension.
pub const M: Dimension = Dimension(0);
/// Columns of the output: the rhs's outer dimension.
pub const N: Dimension = Dimension(1);
/// The contracted dimension.
pub const K: Dimension = Dimension(2);

/// Which block of the contraction a position falls in. Together with [`KI`], where inside that
/// block, it is [`K`], partitioned rather than divided: an operand spanning both addresses the
/// same physical dim, and one spanning `KB` alone, a scales operand, covers a whole block of it.
pub const KB: Dimension = Dimension(16);
/// The position inside a block of [`K`]. See [`KB`].
pub const KI: Dimension = Dimension(17);
/// [`KB`]'s twin on the output's columns: which block of [`N`] a position falls in. A
/// per-channel scale spans `NB` at a block of one.
pub const NB: Dimension = Dimension(18);
/// The position inside a block of [`N`]. See [`NB`].
pub const NI: Dimension = Dimension(19);
/// [`KB`]'s twin on the lhs's rows: which block of [`M`] a position falls in. A per-token
/// scale spans `MB` at a block of one.
pub const MB: Dimension = Dimension(20);
/// The position inside a block of [`M`]. See [`MB`].
pub const MI: Dimension = Dimension(21);

/// Which group of a K-grouped product: the batch dimension of its dense problem, of one batch dim.
pub const GROUP: Dimension = batch_dimension(0);
/// Which of a row's selected experts: a routed matmul's second output dimension.
pub const SLOT: Dimension = Dimension(22);
/// Which expert of a stacked rhs: a routed walk visits the one a row's slot names.
pub const EXPERT: Dimension = Dimension(23);

/// The label of the `index`th batch dim, outermost first.
pub const fn batch_dimension(index: usize) -> Dimension {
    Dimension(3 + index as u8)
}

/// The two dimensions a dimension is: which block of it a position falls in, and where inside that
/// block. A dimension nothing groups still has both: one block over its whole extent.
///
/// # Panics
///
/// On anything but [`M`], [`N`] or [`K`]: the blocks of a block have no meaning.
pub fn blocked(dimension: Dimension) -> (Dimension, Dimension) {
    match dimension {
        M => (MB, MI),
        N => (NB, NI),
        K => (KB, KI),
        other => panic!("{other:?} is not one of the matmul dimensions"),
    }
}

/// The named dimensions: the one list [`dimension_name`] and [`labels`] read.
const NAMED: [(Dimension, &str); 11] = [
    (M, "M"),
    (N, "N"),
    (K, "K"),
    (KB, "KB"),
    (KI, "KI"),
    (MB, "MB"),
    (MI, "MI"),
    (NB, "NB"),
    (NI, "NI"),
    (SLOT, "SLOT"),
    (EXPERT, "EXPERT"),
];

/// The name of a matmul dimension.
///
/// # Panics
///
/// On a batch dimension, which has no name: a table over a space that has one prints it by index
/// ([`labels`]).
pub fn dimension_name(dimension: Dimension) -> &'static str {
    NAMED
        .iter()
        .find(|&&(named, _)| named == dimension)
        .map(|&(_, name)| name)
        .unwrap_or_else(|| panic!("{dimension:?} is not a named matmul dimension"))
}

/// The labels a partitioning over `space` prints under
/// ([`Partitioning::table`](cubecl::tile::Partitioning::table)): every dimension of the space that
/// has a name. A batch dimension prints as its index.
pub fn labels(space: &Space) -> Vec<(Dimension, &'static str)> {
    NAMED
        .iter()
        .copied()
        .filter(|&(dimension, _)| space.contains(dimension))
        .collect()
}
