//! Depthwise convolution, as a direct client of the tile DSL.
//!
//! Every accelerated convolution routine in this crate refuses `groups != 1`, so a depthwise
//! layer autotunes against a single surviving candidate and never reaches an accelerated path at
//! all. This is the missing one, and it is deliberately *not* built on the blueprint/routine
//! machinery the rest of `kernels::forward` uses: a depthwise convolution is not a contraction
//! over channels, so a stage hierarchy sized for one has nothing to size here.
//!
//! What it is instead: a dense convolution contracts input channels into output channels, so the
//! channel pair appears in the two operands and not in the accumulator. A depthwise one has no
//! such pairing — each channel carries its own filter and reaches exactly one output channel — so
//! a single channel axis appears in *all three* operands. That makes it a batch axis, and the
//! contraction is over the window taps alone. Written that way it is a space and a projection,
//! and `Tile::mma` is the whole body.
//!
//! The layout follows from the same fact. Channels stay innermost (NHWC), which is what a
//! bandwidth-bound depthwise pass wants: consecutive lanes read consecutive channels of one pixel
//! and the read coalesces.

use cubecl::{Runtime, prelude::*};
use cubek_tile::*;

use crate::{components::ConvSetupError, launch::ConvolutionArgs};

// Output positions, the channel axis every operand shares, and the window taps. The taps are
// last: the gather leaf lines the input along the fastest contracted axis, so they have to be the
// operand's innermost logical axes even though the innermost *physical* one is the channel.
/// The leaf a memory-backed operand reads through. 16 is the fragment width the memory MMA
/// config walks; nothing here is accelerated, so the flags stay off.
const MEMORY_LEAF: Leaf = Leaf::memory(MemoryMmaConfig::new(16, false, false));

const B: Axis = Axis(0);
const OH: Axis = Axis(1);
const OW: Axis = Axis(2);
const C: Axis = Axis(3);
const RH: Axis = Axis(4);
const RW: Axis = Axis(5);

/// `out[b, oh, ow, c] = Σ_{rh, rw} input[b, oh*sh + rh*dh - ph, ow*sw + rw*dw - pw, c] * w[rh, rw, c]`
///
/// The same body the dense convolution runs. `C` being one of the accumulator's own axes is what
/// makes it batched rather than contracted; the leaf reads that off the spaces.
#[cube(launch)]
fn depthwise_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

/// How much of each axis one cube takes. Spatial tiles stay small because the window overlaps —
/// neighbouring output positions re-read the same input — while the channel tile is what the read
/// coalesces along, so it is the one worth making wide.
const TILE_OH: usize = 4;
const TILE_OW: usize = 4;
const TILE_C: usize = 32;

/// Launch a depthwise convolution.
///
/// Returns [`ConvSetupError::NotDepthwise`] rather than computing a wrong answer when handed a
/// convolution that is not one filter per channel; the tuner reads that as "this candidate does
/// not apply here", which is exactly what it is.
#[allow(clippy::too_many_arguments)]
pub fn launch_depthwise<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorArg<R>,
    weight: TensorArg<R>,
    out: TensorArg<R>,
    in_shape: &[usize],
    weight_shape: &[usize],
    out_shape: &[usize],
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    // NHWC throughout: [batch, h, w, channels].
    let channels = in_shape[3];
    if groups != channels || out_shape[3] != channels {
        return Err(ConvSetupError::NotDepthwise { groups, channels });
    }

    let (b, oh, ow) = (out_shape[0], out_shape[1], out_shape[2]);
    // Burn hands weights as [out_channels, kh, kw, in_channels / groups]; depthwise makes that
    // last axis 1, so the filter is [c, kh, kw] with the channel outermost.
    let (rh, rw) = (weight_shape[1], weight_shape[2]);
    let [sh, sw] = args.stride;
    let [ph, pw] = args.padding;
    let [dh, dw] = args.dilation;

    let space = Tiling::new()
        .extents(&[(B, b), (OH, oh), (OW, ow), (C, channels), (RH, rh), (RW, rw)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(B, Cut::sequential(1))
                .axis(OH, Cut::sequential(TILE_OH))
                .axis(OW, Cut::sequential(TILE_OW))
                .axis(C, Cut::sequential(TILE_C))
                .axis(RH, Cut::sequential(rh))
                .axis(RW, Cut::sequential(rw))
        })
        .build();

    // Two gathered physical axes, one per spatial pair, each carrying its padding as the
    // projection's constant term. Batch and channel ride identity.
    let in_spec = TileSpec::new(
        Projection::new(
            &[B, OH, OW, C, RH, RW],
            &[
                PhysicalAxisMap::of(B),
                PhysicalAxisMap::affine_with_offset(&[(OH, sh), (RH, dh)], -(ph as isize)),
                PhysicalAxisMap::affine_with_offset(&[(OW, sw), (RW, dw)], -(pw as isize)),
                PhysicalAxisMap::of(C),
            ],
        ),
        MEMORY_LEAF,
    )
    // The padded border reads outside the input; `checked` is what makes it read as zero.
    .checked(true);

    let w_spec = TileSpec::direct(&[C, RH, RW], MEMORY_LEAF);
    let out_spec = TileSpec::direct(&[B, OH, OW, C], MEMORY_LEAF);

    depthwise_kernel::launch::<R>(
        client,
        space.cube_count(),
        space.cube_dim(client),
        TileArgLaunch::new(input, in_spec),
        TileArgLaunch::new(weight, w_spec),
        TileArgLaunch::new(out, out_spec),
        space,
        dtype,
    );

    Ok(())
}
