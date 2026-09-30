//! The [`Destination`] seam, through which a kernel stores into [`Output`] or a caller's own
//! argument.

use cubecl::prelude::*;

use crate::{Bound, Output, OutputArgs, Partitioning, Tile, kind::Write};

/// Which launch argument carries a kernel's output and how the kernel serves it as a [`Tile`].
#[cube]
pub trait Destination: Send + Sync + 'static {
    /// The launch argument carrying the output and its spec, at element `E` and width `V`.
    type Arg<E: Numeric, V: Size>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`] under the kernel's one `partitioning`.
    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] partitioning: Partitioning,
    ) -> Tile<E>;
}

/// [`Destination`]'s host half: how what the launch built becomes the argument.
pub trait DestinationLaunch: Destination {
    /// What the argument is made from, before the kernel's element and width type it.
    type Operand;

    fn arg<E: Numeric, V: Size>(
        operand: Self::Operand,
    ) -> <Self::Arg<E, V> as LaunchArg>::RuntimeArg;
}

/// The default [`Destination`]: an [`Output`] buffer.
pub struct Buffered;

#[cube]
impl Destination for Buffered {
    type Arg<E: Numeric, V: Size> = Output<'static, E, V>;

    fn tile<E: Numeric, V: Size>(
        arg: &Self::Arg<E, V>,
        #[comptime] partitioning: Partitioning,
    ) -> Tile<E> {
        arg.tile(partitioning)
    }
}

/// [`Write::Replace`] binds an [`Output::Tensor`], [`Write::Accumulate`] an [`Output::Atomic`].
impl DestinationLaunch for Buffered {
    type Operand = (Bound, Write);

    fn arg<E: Numeric, V: Size>((bound, write): (Bound, Write)) -> OutputArgs<'static, E, V> {
        match write {
            Write::Replace => bound.output(),
            Write::Accumulate => bound.atomic(),
            Write::Relay => panic!(
                "Buffered: a relayed output is opened in the kernel with `TileArg::relay`, which \
                 takes its turn counters beside it; bind the carry as a plain `TileArg`"
            ),
        }
    }
}
