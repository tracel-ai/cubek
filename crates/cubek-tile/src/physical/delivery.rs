//! How an operand's bytes move: the [`Delivery`] (cooperative buffer copy, one storage block
//! at a time, or TMA bulk copy) and its type-level twin [`DeliveryFamily`], which lets one
//! kernel body serve every argument type.

use cubecl::prelude::*;

use crate::{Blocks, Space, StridedOperand, Sync, Tile, TileArg, TmaTileArg};

/// How an operand reaches a stage: a buffered cooperative copy, the same copy of one storage
/// block that is the stage, coordinate-backed cooperative materialization, or a TMA hardware
/// bulk copy. Read off a tile via [`delivery`](crate::Tile::delivery); the staging sync comes
/// from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    #[default]
    Copy,
    /// A block-stored operand whose storage block is a level of the kernel's nest
    /// ([`Blocks`]): the stage is one contiguous run of the buffer, copied cooperatively. The
    /// shape a TMA box is, moved by the cube's units rather than the hardware.
    Block,
    Procedural,
    Tma,
}

/// CUDA caps each TMA box dimension at 256; a bulk copy fills one smem stage, so the
/// stage edges are the box dims.
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// The synchronization required to materialize this source in a staging slot.
    pub(crate) fn rendezvous(&self) -> Sync {
        match self {
            Delivery::Copy | Delivery::Block | Delivery::Procedural => Sync::Cube,
            Delivery::Tma => Sync::Barrier,
        }
    }

    /// Reject a plan the TMA descriptor path can't encode, so a bad plan fails here as a
    /// clean error instead of at descriptor encoding on the driver. `boxes` are the
    /// bulk-copy box dims (one stage per box); `batched` = any surviving batch dim.
    /// A no-op unless this is [`Delivery::Tma`].
    pub fn validate_tma(&self, boxes: &[usize], batched: bool) -> Result<(), String> {
        if !self.is_tma() {
            return Ok(());
        }
        // The descriptor is 3-D `(batch, row, col)`; surviving batch dims need a
        // batch-aware descriptor path not wired yet.
        if batched {
            return Err("TMA: batched problems are not supported yet".to_string());
        }
        if let Some(&max) = boxes.iter().max()
            && max > TMA_MAX_BOX_DIM
        {
            return Err(format!(
                "TMA: box {boxes:?} exceeds the {TMA_MAX_BOX_DIM}-per-axis box limit"
            ));
        }
        Ok(())
    }
}

/// [`Delivery`]'s type-level twin: which launchable argument carries an operand and how a
/// kernel serves that argument as a [`Tile`]. Each argument bundles its own comptime
/// [`TileSpec`] ([`TileArg`] strided or block-stored, [`TmaTileArg`] tensor map), so a tensor
/// can never pair with another operand's spec; only the kernel's one [`Space`] crosses the
/// seam. A kernel body written over `D: DeliveryFamily` runs strided, block-stored or TMA
/// unchanged; the launch entry picks the family. One family covers both operands, since
/// [`Sync::for_deliveries`](crate::Sync::for_deliveries) rejects a mixed pair anyway.
#[cube]
pub trait DeliveryFamily: Send + core::marker::Sync + 'static {
    /// The launchable argument carrying one operand and its spec.
    type Arg<E: Numeric, V: Size>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`]: the kernel's one `space` projected onto the
    /// argument's own spec axes.
    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E>;
}

/// The families whose argument is a plain tensor + its spec ([`TileArg`]): what a built
/// [`StridedOperand`] launches as. Lets a launch entry written over `D` hand its operands to
/// the kernel without naming the erased element types the kernel's launch is spelled in.
pub trait TensorDelivery: DeliveryFamily {
    /// The operand as this family's launch argument.
    fn operand<E: Numeric, V: Size>(
        operand: StridedOperand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg;
}

impl TensorDelivery for Strided {
    fn operand<E: Numeric, V: Size>(
        operand: StridedOperand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg {
        operand.arg::<E, V>()
    }
}

impl TensorDelivery for Block {
    fn operand<E: Numeric, V: Size>(
        operand: StridedOperand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg {
        operand.arg::<E, V>()
    }
}

/// [`Delivery::Copy`]'s family: a plain tensor + spec ([`TileArg`]), tiled in-kernel
/// by [`Tile::of`]. Reads rows, so a block-stored operand is refused here rather than read
/// across its block boundaries; that operand rides [`Block`].
pub struct Strided;

/// [`Delivery::Block`]'s family: the same tensor + spec ([`TileArg`]), the spec stating the
/// level of the kernel's nest its storage block is the tile of ([`Blocks::Above`]). The plain
/// operand beside a block-stored one (an activation beside a packed weight) rides here untiled;
/// only the operand whose spec states a block moves as one.
pub struct Block;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaTileArg`]), hardware bulk-copied.
pub struct Tma;

#[cube]
impl DeliveryFamily for Strided {
    type Arg<E: Numeric, V: Size> = TileArg<'static, E, V>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E> {
        comptime!(match arg.spec.blocks {
            Blocks::Whole => {}
            Blocks::Above(level) => panic!(
                "Strided: this operand arrived block-stored (its block is the tile of level \
                 {level}), which the strided delivery reads across; launch it under Block"
            ),
            Blocks::Held => panic!("Strided: a launched spec is never inside a block"),
        });
        arg.tile(space)
    }
}

#[cube]
impl DeliveryFamily for Block {
    type Arg<E: Numeric, V: Size> = TileArg<'static, E, V>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E> {
        arg.tile(space)
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric, V: Size> = TmaTileArg<E>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E, V>, #[comptime] space: Space) -> Tile<E> {
        arg.tile(space)
    }
}
