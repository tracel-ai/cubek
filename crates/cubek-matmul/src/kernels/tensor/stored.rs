use super::{PackedTensor, PlainTensor, ScaledTensor};
use crate::kernels::select::problem::OperandKind;
use cubecl::client::Client;
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::QuantScheme;
use cubecl::server::BufferBinding;
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
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
    /// 8-bit floats under scales constant along the contraction.
    Scaled(ScaledTensor),
    /// Packed words under their scales.
    Packed(PackedTensor),
}

impl StoredTensor {
    /// A quantized matrix over `dimensions`, in the order its shape states them, as a framework
    /// stores one: `values`, its buffer bound under its logical shape and strides, stored as
    /// `scheme` says, under its `scales` and, where `scheme` has a second level, its
    /// `outer_scales`. 8-bit floats under scales constant along the contraction
    /// ([`ScaledTensor::reads`]), or packed ([`PackedTensor::of_quantized`]).
    ///
    /// # Errors
    ///
    /// As [`ScaledTensor::of_quantized`] and [`PackedTensor::of_quantized`].
    pub fn of_quantized(
        client: &Client,
        values: TensorBinding,
        scheme: QuantScheme,
        scales: PlainTensor,
        outer_scales: Option<PlainTensor>,
        dimensions: [Dimension; 2],
    ) -> Result<Self, Decline> {
        match ScaledTensor::reads(&scheme, dimensions) {
            true => ScaledTensor::of_quantized(client, values, scheme, scales, dimensions)
                .map(Self::Scaled),
            false => {
                PackedTensor::of_quantized(client, values, scheme, scales, outer_scales, dimensions)
                    .map(Self::Packed)
            }
        }
    }

    /// The tensor holding its values: itself, or a packed one's words.
    pub fn values(&self) -> &PlainTensor {
        match self {
            Self::Plain(tensor) => tensor,
            Self::Scaled(tensor) => tensor.values(),
            Self::Packed(tensor) => tensor.words(),
        }
    }

    /// What the tensor is, as a key tells operands apart: float, or quantized under its scheme.
    pub fn kind(&self) -> OperandKind {
        match self {
            Self::Plain(_) => OperandKind::Float,
            Self::Scaled(tensor) => OperandKind::Quantized(tensor.scheme()),
            Self::Packed(tensor) => OperandKind::Quantized(tensor.scheme()),
        }
    }

    /// What `dimension` measures, of the two `dimensions` it states last.
    ///
    /// # Panics
    ///
    /// On a dimension not among them.
    pub fn extent(&self, dimensions: [Dimension; 2], dimension: Dimension) -> usize {
        match self {
            Self::Plain(_) | Self::Scaled(_) => {
                let shape = self.values().logical_shape();
                let position = dimensions
                    .iter()
                    .position(|&own| own == dimension)
                    .unwrap_or_else(|| panic!("{dimension:?} is not one of {dimensions:?}"));
                shape[shape.len() - 2 + position]
            }
            Self::Packed(tensor) => tensor.extent(dimension),
        }
    }

    /// Every buffer it binds, in the order its binding holds them: its values, then its scales, the
    /// finest level first. What a kept binding is rebound to, in place of the
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
        let scales: SmallVec<[&PlainTensor; 2]> = match self {
            Self::Plain(_) => SmallVec::new(),
            Self::Scaled(tensor) => core::iter::once(tensor.scales()).collect(),
            Self::Packed(tensor) => tensor.scales().collect(),
        };
        core::iter::once(self.values()).chain(scales)
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
