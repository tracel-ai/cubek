//! The axis labels of a matmul's space, defined once.
//!
//! A [`Space`] is only meaningful against the labels it was built with, so every matmul, its
//! quantized operands and its storage tiling label the same axes the same way. Batch axes are
//! numbered after the three leading ones ([`batch_axis`]); the blocked axes are numbered past the
//! batch slots, since labels only have to be distinct within the space that uses them.

use cubecl::tile::{Axis, Space};

/// Rows of the output: the lhs's outer dimension.
pub const M: Axis = Axis(0);
/// Columns of the output: the rhs's outer dimension.
pub const N: Axis = Axis(1);
/// The contracted dimension.
pub const K: Axis = Axis(2);

/// Which block of the contraction a position falls in. Together with [`KI`], where inside that
/// block, it is [`K`], partitioned rather than divided: an operand spanning both addresses the
/// same physical dim, and one spanning `KB` alone, a scales operand, covers a whole block of it.
pub const KB: Axis = Axis(16);
/// The position inside a block of [`K`]. See [`KB`].
pub const KI: Axis = Axis(17);
/// [`KB`]'s twin on the output's columns: which block of [`N`] a position falls in. A
/// per-channel scale spans `NB` at a block of one.
pub const NB: Axis = Axis(18);
/// The position inside a block of [`N`]. See [`NB`].
pub const NI: Axis = Axis(19);
/// [`KB`]'s twin on the activation's rows: which block of [`M`] a position falls in. A per-token
/// scale spans `MB` at a block of one.
pub const MB: Axis = Axis(20);
/// The position inside a block of [`M`]. See [`MB`].
pub const MI: Axis = Axis(21);

/// Which group of a K-grouped product: the batch axis of its dense problem, of one batch dim.
pub const GROUP: Axis = batch_axis(0);
/// Which of a row's selected experts: a routed matmul's second output axis.
pub const SLOT: Axis = Axis(22);
/// Which expert of a stacked weight: a routed walk visits the one a row's slot names.
pub const EXPERT: Axis = Axis(23);

/// The label of the `index`th batch dim, outermost first.
pub const fn batch_axis(index: usize) -> Axis {
    Axis(3 + index as u8)
}

/// The two axes an axis is: which block of it a position falls in, and where inside that block.
/// An axis nothing groups still has both: one block over its whole extent.
///
/// # Panics
///
/// On anything but [`M`], [`N`] or [`K`]: the blocks of a block have no meaning.
pub fn blocked(axis: Axis) -> (Axis, Axis) {
    match axis {
        M => (MB, MI),
        N => (NB, NI),
        K => (KB, KI),
        other => panic!("{other:?} is not one of the matmul axes"),
    }
}

/// The named axes: the one list [`axis_name`] and [`labels`] read.
const NAMED: [(Axis, &str); 11] = [
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

/// The name of a matmul axis.
///
/// # Panics
///
/// On a batch axis, which has no name: a table over a space that has one prints it by index
/// ([`labels`]).
pub fn axis_name(axis: Axis) -> &'static str {
    NAMED
        .iter()
        .find(|&&(named, _)| named == axis)
        .map(|&(_, name)| name)
        .unwrap_or_else(|| panic!("{axis:?} is not a named matmul axis"))
}

/// The labels a partitioning over `space` prints under
/// ([`Partitioning::table`](cubecl::tile::Partitioning::table)): every axis of the space that has a
/// name. A batch axis prints as its index.
pub fn labels(space: &Space) -> Vec<(Axis, &'static str)> {
    NAMED
        .iter()
        .copied()
        .filter(|&(axis, _)| space.contains(axis))
        .collect()
}
