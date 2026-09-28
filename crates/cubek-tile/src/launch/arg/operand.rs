//! What a kernel binds a global operand as: [`Input`] for the operands it reads, [`Output`] for
//! the one it leaves by. Each variant is named for what is bound, and the variant is comptime: a
//! kernel over `&Input` compiles once per variant, to the same code a kernel over the variant's
//! argument alone would.
//!
//! Who *moves* a bound operand into a stage is a separate statement ([`Delivery`]): a tensor map
//! is only ever bulk-copied, but a plain tensor may be moved more than one way.

use cubecl::prelude::*;

use crate::{
    AccumulateArg, AccumulateArgLaunch, Bound, Partitioning, Storage, Tile, TileArg, TileSpec,
    TmaOperand, TmaTileArg, TmaTileArgLaunch,
};

/// An operand the kernel reads.
///
/// | Variant | Bound as | Read |
/// |---|---|---|
/// | [`Tensor`](Input::Tensor) | a tensor and its spec ([`TileArg`]) | by the cube's units, or directly |
/// | [`TensorMap`](Input::TensorMap) | a TMA tensor map ([`TmaTileArg`]) | bulk-copied into a stage |
#[derive(CubeType, CubeLaunch)]
pub enum Input<'a, E: Numeric, V: Size> {
    Tensor(TileArg<'a, E, V>),
    TensorMap(TmaTileArg<E>),
}

/// The operand a kernel leaves by.
///
/// | Variant | Bound as | Read | Written |
/// |---|---|---|---|
/// | [`Tensor`](Output::Tensor) | a tensor and its spec ([`TileArg`]) | yes: an accumulator may start from it | replaced |
/// | [`Atomic`](Output::Atomic) | an atomic tensor ([`AccumulateArg`]) | never: it arrives holding the identity | added into |
///
/// [`Atomic`](Output::Atomic) is what lets the cubes sharing a cell each hold a slice of it: none
/// reads the cell, every one adds its partial.
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
    /// Serve this operand as a [`Tile`] under the kernel's one `partitioning`: one a drain replaces
    /// or adds into, as the variant says.
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

/// A bound tensor's spec is the operand's own: it lies in its buffer strided or storage-tiled,
/// never already inside one of its storage tiles, which is what a stage's view of it is.
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

    /// This operand as an [`Output::Atomic`]: added into, never read. The buffer has to arrive
    /// holding the monoid's identity.
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
