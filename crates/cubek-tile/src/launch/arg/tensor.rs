//! One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].

use cubecl::prelude::*;

use crate::*;

/// One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric, V: Size> {
    pub tensor: &'a Tensor<Vector<E, V>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric, V: Size> TileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`] over the kernel's one `partitioning`.
    pub fn tile(&self, partitioning: &Partitioning) -> Tile<E> {
        let space = comptime!(partitioning.clone());
        GlobalOperand::<E>::tensor(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile) reading `O` out of the stored `E`, unpacked where the binding is
    /// [`packed`](TileSpec::packed).
    pub fn tile_as<O: Numeric>(&self, partitioning: &Partitioning) -> Tile<O> {
        let space = comptime!(partitioning.clone());
        GlobalOperand::<O>::stored(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile) for a partly runtime gather map, its dynamic values in
    /// [`GlobalOperand::gathered`]'s order.
    pub fn tile_gathered(
        &self,
        partitioning: &Partitioning,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<E> {
        let space = comptime!(partitioning.clone());
        GlobalOperand::<E>::gathered(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            coefficients,
            offsets,
        )
        .tile(comptime!(space.levels().to_vec()))
    }
}
