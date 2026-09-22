//! One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].

use cubecl::prelude::*;

use crate::*;

/// One strided operand as a single launch argument: the plain tensor (its element type carrying
/// the served width) paired with its comptime [`TileSpec`], so a tensor is never launched against
/// another's spec. The one [`Space`] arrives on its own; [`tile`](TileArg::tile) projects it.
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric, V: Size> {
    pub tensor: &'a Tensor<Vector<E, V>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric, V: Size> TileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`]: the kernel's one `space` projected onto this
    /// operand's `spec` axes, under the levels that partition it.
    pub fn tile(&self, #[comptime] space: Partitioning) -> Tile<E> {
        Tile::<E>::of(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .under(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](Self::tile) with the element stated instead of inferred: `E` is what the binding
    /// *stores* and `O` what the tile reads out of it, unpacked where the binding states a
    /// [`packing`](TileSpec::packed); a packed element is the word, so `O` cannot be read off it.
    ///
    /// This is what a kernel writes when how its operand is stored is the binding's business and
    /// not its own: a factor packed into words and a factor lying at its own element are the same
    /// call.
    pub fn tile_as<O: Numeric>(&self, #[comptime] space: Partitioning) -> Tile<O> {
        Tile::<O>::of_stored(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .under(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](Self::tile) for a partly runtime gather map: `coefficients` holds one value per
    /// [`Scale::Dynamic`](crate::Scale) term and [`Divisor::Dynamic`](crate::Divisor) axis, and
    /// `offsets` one per [`Offset::Dynamic`](crate::Offset), in [`Tile::of_gathered`]'s order.
    pub fn tile_gathered(
        &self,
        #[comptime] space: Partitioning,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<E> {
        Tile::<E>::of_gathered(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            coefficients,
            offsets,
        )
        .under(comptime!(space.levels().to_vec()))
    }
}
