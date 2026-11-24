use super::{PackedTensor, PlainTensor, RowScaledTensor};
use crate::kernels::select::problem::OperandKind;
use cubecl::server::BufferBinding;
use cubecl::tile::Dimension;
use cubecl::zspace::SmallVec;
use cubecl::zspace::metadata::Metadata;

/// A tensor as it is stored, which is how a kernel reads it: its values where they lie, or packed
/// under its scales. Each binds as one tile, so a kernel reads whichever it is handed.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per product's tensors, moved once into them and read there"
)]
pub enum StoredTensor {
    /// Its values as they are.
    Plain(PlainTensor),
    /// Packed words under their scales.
    Packed(PackedTensor),
    /// 8-bit float values under one scale a row, which come out of the sum.
    RowScaled(RowScaledTensor),
}

impl StoredTensor {
    /// The tensor holding its values: itself, or a packed one's words.
    pub fn values(&self) -> &PlainTensor {
        match self {
            Self::Plain(tensor) => tensor,
            Self::Packed(tensor) => tensor.words(),
            Self::RowScaled(tensor) => tensor.values(),
        }
    }

    /// What the tensor is, as a key tells operands apart: float, or quantized under its scheme.
    pub fn kind(&self) -> OperandKind {
        match self {
            Self::Plain(_) => OperandKind::Float,
            Self::Packed(tensor) => OperandKind::Quantized(tensor.scheme()),
            Self::RowScaled(tensor) => OperandKind::Quantized(tensor.scheme()),
        }
    }

    /// What `dimension` measures, of the two `dimensions` it states last.
    ///
    /// # Panics
    ///
    /// On a dimension not among them.
    pub fn extent(&self, dimensions: [Dimension; 2], dimension: Dimension) -> usize {
        match self {
            Self::Plain(_) | Self::RowScaled(_) => {
                let shape = self.values().shape();
                let position = dimensions
                    .iter()
                    .position(|&own| own == dimension)
                    .unwrap_or_else(|| panic!("{dimension:?} is not one of {dimensions:?}"));
                shape[shape.rank() - 2 + position]
            }
            Self::Packed(tensor) => tensor.extent(dimension),
        }
    }

    /// Every buffer it binds, in the order its binding holds them: its values, then a packed one's
    /// block scales and its outer scale, or a row-scaled one's scales. What a kept binding is
    /// rebound to, in place of the buffers it was made over.
    pub fn buffers(&self) -> Vec<BufferBinding> {
        self.tensors()
            .map(|tensor| tensor.binding.handle.clone())
            .collect()
    }

    /// The bytes it occupies, what a kernel streaming it at its stored width reads: the union of
    /// the regions its [`buffers`](Self::buffers) name, allocation by allocation, rather than the
    /// buffers those regions lie in.
    ///
    /// Not their sum either: a framework stores a packed tensor's words, block scales and
    /// per-tensor scale in one buffer, and where the words' region spans the scales a sum counts
    /// the scales twice.
    pub fn bytes(&self) -> usize {
        union_bytes(
            self.tensors()
                .map(|tensor| tensor.binding.handle.claim_key()),
        ) as usize
    }

    /// How each of [`buffers`](Self::buffers) lies, in its order: what a binding derived over them
    /// reads of them, so a kept binding is reused only over buffers that lie the same.
    pub fn layouts(&self) -> impl Iterator<Item = Metadata> + '_ {
        self.tensors().map(PlainTensor::meta)
    }

    /// The tensors behind [`buffers`](Self::buffers), in its order.
    fn tensors(&self) -> impl Iterator<Item = &PlainTensor> {
        let (values, packed, row) = match self {
            Self::Plain(tensor) => (tensor, None, None),
            Self::Packed(tensor) => (tensor.words(), Some(tensor.scales()), None),
            Self::RowScaled(tensor) => (tensor.values(), None, Some(tensor.scales())),
        };
        core::iter::once(values)
            .chain(packed.into_iter().flatten())
            .chain(row)
    }
}

/// Bytes the union of `ranges` covers, each `(allocation, start, end)` as a binding's claim key
/// states its range: overlapping ranges of one allocation count once, ranges of different
/// allocations each count. A tensor binds a few buffers, so they are merged where they stand.
///
/// A free function rather than a method of [`StoredTensor`]: it reads no tensor, and the
/// allocation's identity is a parameter so the merge is tested without a device.
fn union_bytes<Allocation: Ord + Copy>(
    ranges: impl IntoIterator<Item = (Allocation, u64, u64)>,
) -> u64 {
    let mut ranges: SmallVec<[(Allocation, u64, u64); 4]> = ranges.into_iter().collect();
    ranges.sort_unstable();
    let mut bytes = 0;
    let mut covered: Option<(Allocation, u64)> = None;
    for (allocation, start, end) in ranges {
        let reached = match covered {
            Some((same, reached)) if same == allocation => reached,
            _ => start,
        };
        bytes += end.saturating_sub(start.max(reached));
        covered = Some((allocation, reached.max(end)));
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::union_bytes;

    /// A packed weight's words view the whole allocation, its block scales and per-tensor scale
    /// views inside it: the allocation counts once.
    #[test]
    fn views_of_one_allocation_count_once() {
        assert_eq!(
            union_bytes([(0, 0, 1000), (0, 800, 990), (0, 990, 994)]),
            1000
        );
    }

    /// Disjoint views of one allocation, and views of two allocations, each count.
    #[test]
    fn disjoint_ranges_each_count() {
        assert_eq!(union_bytes([(0, 0, 100), (0, 200, 300), (1, 0, 50)]), 250);
    }

    /// Partly overlapping views count their union, whatever order they are bound in.
    #[test]
    fn overlapping_ranges_count_their_union() {
        assert_eq!(union_bytes([(0, 50, 150), (0, 0, 100)]), 150);
    }
}
