use cubecl::tile::Axis;

/// A batch.
pub const B: Axis = Axis(0);
/// A key head, which a group of query heads shares.
pub const KV: Axis = Axis(1);
/// A query head of the group sharing one key head.
pub const G: Axis = Axis(2);
/// A query position. It lies beside the keys, so a score's rows are its queries: an owner's box
/// holds rows of one head, and its heads are walked.
pub const Q: Axis = Axis(3);
/// A key position: the column of `query · keyᵀ` and the contracted axis of `p · value`.
pub const S: Axis = Axis(4);
/// The head dim `query · keyᵀ` contracts.
pub const D: Axis = Axis(5);
/// The value dim, the output's column.
pub const DV: Axis = Axis(6);
