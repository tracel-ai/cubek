use super::{PackedTensor, PlainTensor};
use cubecl::server::BufferBinding;
use cubecl::tile::Axis;
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
