use cubecl::tile::Dimension;

/// A batch.
pub const B: Dimension = Dimension(0);
/// A key head, which a group of query heads shares.
pub const KV: Dimension = Dimension(1);
/// A query head of the group sharing one key head.
pub const G: Dimension = Dimension(2);
/// A query position. It lies beside the keys, so a score's rows are its queries: an owner's box
/// holds rows of one head, and its heads are walked.
pub const Q: Dimension = Dimension(3);
/// A key position: the column of `query · keyᵀ` and the contracted dimension of `p · value`.
pub const S: Dimension = Dimension(4);
/// The head dim `query · keyᵀ` contracts.
pub const D: Dimension = Dimension(5);
/// The value dim, the output's column.
pub const DV: Dimension = Dimension(6);
