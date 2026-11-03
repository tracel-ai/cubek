use super::{PackedTensor, PlainTensor};
use crate::kernels::select::problem::OperandKind;
use cubecl::server::BufferBinding;
use cubecl::tile::Axis;
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
}

impl StoredTensor {
    /// The tensor holding its values: itself, or a packed one's words.
    pub fn values(&self) -> &PlainTensor {
        match self {
            Self::Plain(tensor) => tensor,
            Self::Packed(tensor) => tensor.words(),
        }
    }

    /// What the tensor is, as a key tells operands apart: float, or quantized under its scheme.
    pub fn kind(&self) -> OperandKind {
        match self {
            Self::Plain(_) => OperandKind::Float,
            Self::Packed(tensor) => OperandKind::Quantized(tensor.scheme()),
        }
    }

    /// What `axis` measures, of the two `axes` it states last.
    ///
    /// # Panics
    ///
    /// On an axis not among them.
    pub fn extent(&self, axes: [Axis; 2], axis: Axis) -> usize {
        match self {
            Self::Plain(tensor) => {
                let shape = tensor.shape();
                let position = axes
                    .iter()
                    .position(|&own| own == axis)
                    .unwrap_or_else(|| panic!("{axis:?} is not one of {axes:?}"));
                shape[shape.rank() - 2 + position]
            }
            Self::Packed(tensor) => tensor.extent(axis),
        }
    }

    /// Every buffer it binds, in the order its binding holds them: its values, then a packed one's
    /// block scales and its per-tensor scale. What a kept binding is rebound to, in place of the
    /// buffers it was made over ([`TileArgLaunch::rebound`](cubecl::tile::TileArgLaunch::rebound)).
    pub fn buffers(&self) -> Vec<BufferBinding> {
        self.tensors()
            .map(|tensor| tensor.binding.handle.clone())
            .collect()
    }

    /// The bytes it occupies, what a kernel streaming it at its stored width reads: the union of
    /// the ranges its [`buffers`](Self::buffers) view, allocation by allocation.
    ///
    /// Not their sum: a packed tensor's words, block scales and per-tensor scale are views of one
    /// allocation, the words' spanning the scales, and a binding's `size` is the whole
    /// allocation's, so a sum counts an NVFP4 weight about three times over.
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
        let (values, scales) = match self {
            Self::Plain(tensor) => (tensor, None),
            Self::Packed(tensor) => (tensor.words(), Some(tensor.scales())),
        };
        core::iter::once(values).chain(scales.into_iter().flatten())
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
