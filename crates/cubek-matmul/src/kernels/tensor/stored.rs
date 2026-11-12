use super::{PackedTensor, PlainTensor, ScaledTensor};
use crate::kernels::select::problem::OperandKind;
use cubecl::server::BufferBinding;
use cubecl::tile::Axis;
use cubecl::zspace::SmallVec;
use cubecl::zspace::metadata::Metadata;

/// A tensor as it is stored, which is how a kernel reads it: its values where they lie, packed
/// under its scales, or multiplied by a scale tensor as they are read. Each binds as one tile, so a
/// kernel reads whichever it is handed.
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
    /// Float values multiplied by a scale tensor as a stage is filled.
    Scaled(ScaledTensor),
}

impl StoredTensor {
    /// The tensor holding its values: itself, or a packed one's words.
    pub fn values(&self) -> &PlainTensor {
        match self {
            Self::Plain(tensor) => tensor,
            Self::Packed(tensor) => tensor.words(),
            Self::Scaled(tensor) => tensor.values(),
        }
    }

    /// The tensor holding its values where they are floats, which a scale may multiply: none for
    /// a packed one, whose words are not values.
    pub fn float(&self) -> Option<&PlainTensor> {
        match self {
            Self::Plain(tensor) => Some(tensor),
            Self::Scaled(tensor) => Some(tensor.values()),
            Self::Packed(_) => None,
        }
    }

    /// This float tensor with its values replaced by `values`, which a launch read from them: a
    /// scaled one keeps its scale.
    ///
    /// # Panics
    ///
    /// On a packed tensor, whose words are not values.
    pub(crate) fn over(self, client: &cubecl::client::Client, values: PlainTensor) -> Self {
        match self {
            Self::Plain(_) => Self::Plain(values),
            Self::Scaled(tensor) => Self::Scaled(tensor.over(client, values)),
            Self::Packed(_) => panic!("a packed tensor's words are not replaced by values"),
        }
    }

    /// What the tensor is, as a key tells operands apart: float, or quantized under its scheme.
    pub fn kind(&self) -> OperandKind {
        match self {
            Self::Plain(_) => OperandKind::Float,
            Self::Packed(tensor) => OperandKind::Quantized(tensor.scheme()),
            Self::Scaled(tensor) => OperandKind::Scaled(tensor.scale().elem),
        }
    }

    /// What `axis` measures, of the two `axes` it states last.
    ///
    /// # Panics
    ///
    /// On an axis not among them.
    pub fn extent(&self, axes: [Axis; 2], axis: Axis) -> usize {
        if let Self::Packed(tensor) = self {
            return tensor.extent(axis);
        }
        let shape = self.values().shape();
        let position = axes
            .iter()
            .position(|&own| own == axis)
            .unwrap_or_else(|| panic!("{axis:?} is not one of {axes:?}"));
        shape[shape.rank() - 2 + position]
    }

    /// Every buffer it binds, in the order its binding holds them: its values, then a packed one's
    /// block scales and its per-tensor scale. What a kept binding is rebound to, in place of the
    /// buffers it was made over.
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
        let packed_scales = match self {
            Self::Packed(tensor) => Some(tensor.scales()),
            Self::Plain(_) | Self::Scaled(_) => None,
        };
        let scale = match self {
            Self::Scaled(tensor) => Some(tensor.scale()),
            Self::Plain(_) | Self::Packed(_) => None,
        };
        core::iter::once(self.values())
            .chain(packed_scales.into_iter().flatten())
            .chain(scale)
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
