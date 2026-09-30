//! What a kernel binds a global operand as: [`Input`] or [`Output`].

use cubecl::prelude::*;

use crate::{
    AccumulateArg, AccumulateArgLaunch, Bound, Partitioning, Storage, Tile, TileArg, TileSpec,
    TmaOperand, TmaTileArg, TmaTileArgLaunch,
};

/// An operand the kernel reads: a tensor ([`TileArg`]) or a TMA tensor map ([`TmaTileArg`]).
#[derive(CubeType, CubeLaunch)]
pub enum Input<'a, E: Numeric, V: Size> {
    Tensor(TileArg<'a, E, V>),
    TensorMap(TmaTileArg<E>),
}

/// The operand a kernel leaves by: a tensor read and replaced ([`TileArg`]), or an atomic tensor
/// added into and never read ([`AccumulateArg`]).
#[derive(CubeType, CubeLaunch)]
pub enum Output<'a, E: Numeric, V: Size> {
    Tensor(TileArg<'a, E, V>),
    Atomic(AccumulateArg<'a, E>),
}

#[cube]
impl<'a, E: Numeric, V: Size> Input<'a, E, V> {
    /// Serve this operand as a [`Tile`] under the kernel's one `partitioning`.
    pub fn tile(&self, #[comptime] partitioning: Partitioning) -> Tile<E> {
        match self {
            Input::Tensor(arg) => {
                comptime!(tensor_spec_is_launched(&arg.spec));
                arg.tile(partitioning)
            }
            Input::TensorMap(arg) => arg.tile(partitioning),
        }
    }
}

#[cube]
impl<'a, E: Numeric, V: Size> Output<'a, E, V> {
    /// Serve this operand as a [`Tile`] under the kernel's one `partitioning`.
    pub fn tile(&self, #[comptime] partitioning: Partitioning) -> Tile<E> {
        match self {
            Output::Tensor(arg) => {
                comptime!(tensor_spec_is_launched(&arg.spec));
                arg.tile(partitioning)
            }
            Output::Atomic(arg) => arg.tile::<V>(partitioning),
        }
    }
}

/// Panics on a spec that is already inside a storage tile.
fn tensor_spec_is_launched(spec: &TileSpec) {
    match spec.storage {
        Storage::Strided | Storage::Tiled(_) => {}
        Storage::Contiguous => panic!("a launched spec is never inside a storage tile"),
    }
}

impl Bound {
    /// This operand as an [`Input::Tensor`].
    pub fn input<E: Numeric, V: Size>(self) -> InputArgs<'static, E, V> {
        InputArgs::Tensor(self.arg())
    }

    /// This operand as an [`Output::Tensor`]: read and replaced.
    pub fn output<E: Numeric, V: Size>(self) -> OutputArgs<'static, E, V> {
        OutputArgs::Tensor(self.arg())
    }

    /// This operand as an [`Output::Atomic`]; the buffer must arrive holding zero, since it only
    /// ever adds.
    pub fn atomic<E: Numeric, V: Size>(self) -> OutputArgs<'static, E, V> {
        let spec = self.spec.clone();
        OutputArgs::Atomic(AccumulateArgLaunch::new(self.tensor(), spec))
    }
}

impl TmaOperand {
    /// This tensor map as an [`Input::TensorMap`].
    pub fn input<E: Numeric, V: Size>(self) -> InputArgs<'static, E, V> {
        InputArgs::TensorMap(TmaTileArgLaunch::tensor_map(
            self.map, &self.axes, self.shape,
        ))
    }
}
