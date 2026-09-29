//! The seam for an output this crate does not know: a [`Destination`] names the launch argument a
//! kernel leaves by and serves it as a [`Tile`], so a kernel generic over one stores into
//! [`Output`] or into a caller's own argument with the same body.
//!
//! [`Output`] is the default and covers every buffer ([`Buffered`]). A caller implements the pair
//! for a destination with no buffer behind it, a fused epilogue's sink being the one it exists
//! for: its argument carries whatever the store needs, and its `tile` builds a
//! [`GlobalOperand::sink`](crate::kind::GlobalOperand::sink) over an
//! [`ErasedTensor`](cubecl::std::tensor::ErasedTensor) that only writes.
//!
//! Only outputs have the seam. What a kernel reads is a buffer or a tensor map ([`Input`]), and a
//! fused read is an [`ErasedTensor`](cubecl::std::tensor::ErasedTensor) the kernel builds itself.

use cubecl::prelude::*;

use crate::{Bound, Output, OutputArgs, Partitioning, Tile, kind::Write};

/// Which launch argument carries a kernel's output and how the kernel serves it as a [`Tile`].
/// The argument bundles its comptime spec; only the kernel's one [`Partitioning`] crosses the
/// seam.
///
/// A kernel takes `out: &O::Arg<E, V>` and opens it with `O::tile::<E, V>(out, partitioning)`;
/// the launch picks `O`. A family rather than the argument type itself because the element and
/// width are the kernel's generics, often defined at the launch, which the caller cannot name.
///
/// The tile is a destination: a drain replaces or adds into it. Whether it can be read is the
/// implementation's own fact, so a kernel that seeds its accumulator from the output needs one
/// that can ([`Output::Tensor`]); one that drains a finished sum asks nothing a sink lacks.
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

/// The default [`Destination`]: an [`Output`], a buffer written through its address, replaced
/// ([`Output::Tensor`]) or added into ([`Output::Atomic`]).
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

/// A bound output and what its writes do: [`Write::Replace`] binds an [`Output::Tensor`],
/// [`Write::Accumulate`] an [`Output::Atomic`].
impl DestinationLaunch for Buffered {
    type Operand = (Bound, Write);

    fn arg<E: Numeric, V: Size>((bound, write): (Bound, Write)) -> OutputArgs<'static, E, V> {
        match write {
            Write::Replace => bound.output(),
            Write::Accumulate => bound.atomic(),
        }
    }
}
