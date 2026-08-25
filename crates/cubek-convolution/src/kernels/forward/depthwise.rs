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
//! Everything about the layout follows from that one axis. Channels stay innermost (NHWC) and are
//! what the lanes are spent on, so consecutive lanes read consecutive channels of one pixel and
//! the read coalesces; and they are what every operand is *lined* along, so one instruction moves
//! a lane's whole cell. A depthwise pass has too little arithmetic per byte to be anything but
//! bandwidth-bound, and both of those are what let it reach the bandwidth.

use cubecl::{
    Runtime,
    prelude::*,
    server::LaunchError,
    zspace::{Shape, Strides},
};
use cubek_std::InputBinding;
use cubek_tile::*;

use crate::{components::ConvSetupError, launch::ConvolutionArgs};

/// What runs on the cells the last level cuts out. 64 is the register budget in scalars, which
/// at four channels to a line is the same sixteen cells every tiling here blocks into;
/// `split_edge` is what keeps the padded border's guard off the instances that do not straddle
/// it, which for a window this wide is most of them. Nothing here is accelerated, so the fan-out
/// flag stays off: the lines run along the channel, not along `K`.
const INSTRUCTION: Instruction = Instruction::Registers {
    config: RegisterBlock::new(64, true, false),
};

// Output positions, the channel axis every operand shares, and the window taps.
const B: Axis = Axis(0);
const OH: Axis = Axis(1);
const OW: Axis = Axis(2);
const C: Axis = Axis(3);
const RH: Axis = Axis(4);
const RW: Axis = Axis(5);

/// `out[b, oh, ow, c] = Σ_{rh, rw} w[rh, rw, c] · input[b, oh*sh + rh*dh - ph, ow*sw + rw*dw - pw, c]`
///
/// The same body the dense convolution runs. `C` being one of the accumulator's own axes is what
/// makes it batched rather than contracted; the leaf reads that off the spaces.
///
/// The filter is the *lhs* and the map the rhs, which is not arbitrary: the leaf serves the rhs in
/// lines along the accumulator's innermost axis, and that axis is the channel. The filter follows
/// it there — one filter value per channel of the cell — which is what a batched contraction
/// needs and what `V > 1` is.
#[cube(launch)]
fn depthwise_kernel<E: Numeric, V: Size>(
    weight: &TileArg<'_, E, V>,
    input: &TileArg<'_, E, V>,
    out: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let weight = weight.tile(comptime!(space.clone()));
    let input = input.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&weight, &input);
}

/// How one cube's share of the output is shaped, and how the cube's threads divide it.
///
/// The three numbers are three different jobs, which is why they are not one "tile size":
///
/// - `rows` is the plane count. One plane per output row, so it is what fills the cube.
/// - `cols` is the accumulator block one lane keeps in registers. Every column of it re-reads
///   the same filter and overlapping input, so it is what amortises both.
/// - `chans` is how many channel *lines* one lane owns. Lanes are dealt lines interleaved, so
///   whatever this is, consecutive lanes still read consecutive channels and the read coalesces.
/// - `lines` is how many channels one of those lines covers — the width every operand is served
///   in. It is the one knob that trades the two things a depthwise pass is limited by against
///   each other, which is why it is stated and not derived: a wider line is fewer instructions
///   per channel, and also more registers per lane and a wider channel tile, so fewer lanes with
///   anything to do when the block is narrow. It is a ceiling, not a demand — the launch drops to
///   what the buffers can actually be served in.
///
/// The cube's channel tile is `plane_size · lines · chans`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthwiseTiling {
    pub rows: usize,
    pub cols: usize,
    pub chans: usize,
    pub lines: usize,
}

impl Default for DepthwiseTiling {
    /// Four planes of one row each, four output columns per lane, scalar channels.
    ///
    /// Small on both spatial axes on purpose: the window overlaps, so a cube's halo is what it
    /// re-reads, but a *wide* tile is also what pushes its far corner past the padded border and
    /// costs every instance in it the guarded walk. Four each is where the two turned over on
    /// the shapes an encoder actually runs, and it is within 2% of the best fixed tile on every
    /// one of them. The line width is the knob worth deciding per problem, which
    /// [`for_problem`](Self::for_problem) is.
    fn default() -> Self {
        Self {
            rows: 4,
            cols: 4,
            chans: 1,
            lines: 1,
        }
    }
}

impl DepthwiseTiling {
    /// The tiling to run a problem of this shape under.
    ///
    /// Only [`lines`](Self::lines) is decided here, and it is close to a single question: is this
    /// convolution short of instructions or short of bandwidth? A wide line is four times fewer
    /// instructions per channel, and also four times the registers per lane and a four-times
    /// wider channel tile. A deep window over a wide block is instruction-bound and takes the
    /// trade; everything else is already reading memory as fast as the device will read it, and
    /// pays the registers for nothing.
    ///
    /// Measured over EfficientNet-B4's fourteen distinct depthwise shapes at batch 4, on one
    /// RTX 3070 laptop: 16.6 ms of depthwise convolution against 17.3 ms for always-scalar and
    /// 19.3 ms for always-wide, and within 6% of picking the whole tiling per shape. The
    /// thresholds are where the sweep turned over, not a model of the hardware — a different
    /// device is a reason to re-run the sweep, and the benchmark catalogue is what re-runs it.
    pub fn for_problem(channels: usize, taps: usize, lanes: usize) -> Self {
        let deep_window = taps >= 25;
        let wide_block = channels >= 8 * lanes;

        Self {
            lines: match deep_window && wide_block {
                true => 4,
                false => 1,
            },
            ..Default::default()
        }
    }

    /// The space this tiling implies for a problem of these extents.
    ///
    /// Two levels. The first separates the output across the launch grid — an all-`sequential`
    /// level would put the whole convolution in one instance, which is a correct kernel and a
    /// useless one. The second separates what one cube took across the cube's own threads: rows
    /// go to planes, channels to lanes. The taps stay `sequential` throughout — they are the
    /// contraction, and every tap of one output position accumulates into the same register.
    #[allow(clippy::too_many_arguments)]
    fn space(&self, geometry: &Geometry, lanes: usize, width: usize) -> Space {
        let Self {
            rows, cols, chans, ..
        } = *self;
        let Geometry {
            b,
            oh,
            ow,
            c,
            rh,
            rw,
            ..
        } = *geometry;
        let tile_c = lanes * width * chans;

        Tiling::new()
            .extents(&[(B, b), (OH, oh), (OW, ow), (C, c), (RH, rh), (RW, rw)])
            // The channel axis takes X so that the fastest-moving cube index is the one memory is
            // contiguous along.
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(C, Cut::cube(CubeAxis::X, tile_c))
                    .axis(OW, Cut::cube(CubeAxis::Y, cols))
                    .axis(OH, Cut::cube(CubeAxis::Z, rows))
                    .axis(B, Cut::cube(CubeAxis::Z, 1))
                    .axis(RH, Cut::sequential(rh))
                    .axis(RW, Cut::sequential(rw))
            })
            // Rows across the cube's planes, channels across each plane's lanes. Columns stay
            // sequential: they are the register block, not a split.
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(C, interleaved_lanes(width))
                    .axis(OW, Cut::sequential(cols))
                    .axis(OH, Cut::plane(1))
                    .axis(B, Cut::sequential(1))
                    .axis(RH, Cut::sequential(rh))
                    .axis(RW, Cut::sequential(rw))
            })
            .build()
            .with_instruction(INSTRUCTION)
            .resolve_lanes(lanes)
    }
}

/// One line of channels per lane, dealt round-robin, so a lane holding several takes every
/// `plane_size`-th line rather than a contiguous run of them.
///
/// [`Cut::unit`] deals contiguous runs, which puts a stride between what neighbouring lanes read
/// and breaks the coalescing the whole NHWC layout is for. Taking turns instead keeps lane `i` on
/// line `i` of every round, so a round is one contiguous stretch of memory however many rounds
/// there are.
fn interleaved_lanes(width: usize) -> Cut {
    Cut::new(
        width,
        Distribution::Spatial {
            scope: ComputeScope::Unit,
            spread: Spread::Interleaved,
            coverage: Coverage::PlaneLanes,
        },
    )
}

/// The problem, in the terms the space is built from. NHWC throughout.
struct Geometry {
    b: usize,
    oh: usize,
    ow: usize,
    c: usize,
    rh: usize,
    rw: usize,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}

/// Launch a depthwise convolution, tiled as [`DepthwiseTiling::for_problem`] has it.
///
/// Returns [`ConvSetupError::NotDepthwise`] rather than computing a wrong answer when handed a
/// convolution that is not one filter per channel; the tuner reads that as "this candidate does
/// not apply here", which is exactly what it is.
#[allow(clippy::too_many_arguments)]
pub fn launch_depthwise<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    weight: TensorBinding<R>,
    out: TensorBinding<R>,
    in_shape: &[usize],
    weight_shape: &[usize],
    out_shape: &[usize],
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    launch_depthwise_tiled(
        client,
        input,
        weight,
        out,
        in_shape,
        weight_shape,
        out_shape,
        args,
        groups,
        dtype,
        DepthwiseTiling::for_problem(
            in_shape[3],
            weight_shape[1] * weight_shape[2],
            client.properties().hardware.plane_size_max as usize,
        ),
    )
}

/// [`launch_depthwise`] with the tiling stated rather than defaulted. The benchmark catalogue
/// sweeps it; the tuner does not, yet.
#[allow(clippy::too_many_arguments)]
pub fn launch_depthwise_tiled<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    weight: TensorBinding<R>,
    out: TensorBinding<R>,
    in_shape: &[usize],
    weight_shape: &[usize],
    out_shape: &[usize],
    args: ConvolutionArgs<2>,
    groups: usize,
    dtype: ElemType,
    tiling: DepthwiseTiling,
) -> Result<(), ConvSetupError> {
    // NHWC throughout: [batch, h, w, channels].
    let channels = in_shape[3];
    if groups != channels || out_shape[3] != channels {
        return Err(ConvSetupError::NotDepthwise { groups, channels });
    }

    let geometry = Geometry {
        b: out_shape[0],
        oh: out_shape[1],
        ow: out_shape[2],
        c: channels,
        // Burn hands weights as [out_channels, kh, kw, in_channels / groups]; depthwise makes
        // that last axis 1, so the filter is [c, kh, kw] with the channel outermost.
        rh: weight_shape[1],
        rw: weight_shape[2],
        stride: args.stride,
        padding: args.padding,
        dilation: args.dilation,
    };

    // The filter, re-laid so the channel is its innermost dim like every other operand's. It has
    // to be: the leaf serves a cell in lines along the channel, and one filter value broadcast
    // over a line would give every channel of it the first channel's filter. This is the one
    // copy the routine makes, and it is the smallest tensor in the problem by three orders of
    // magnitude — a 1632-channel 5x5 filter is 163 KB against 60 MB of map.
    let weight = channels_innermost(client, weight, &geometry, dtype)
        .map_err(|_| ConvSetupError::Unknown)?;

    let lanes = client.properties().hardware.plane_size_max as usize;
    let width = line_width(
        client,
        channels,
        dtype,
        tiling.lines,
        &[&input, &weight, &out],
    );
    let space = tiling.space(&geometry, lanes, width);

    // A tile that does not divide its axis leaves the last cube short, and a short cube's
    // terminal tile is still the full comptime size — so the cells past the end are addressed and
    // have to be guarded. Per axis, because the guard is real work per access and the axes that
    // need one are rarely the same: a 48x48 map divides evenly by any tile here while a
    // 24-channel block never fills one lane-width.
    let guard = |ragged: bool| ragged.then_some(Boundary::Zero);
    let ragged_c = !channels.is_multiple_of(lanes * width * tiling.chans);
    let ragged_oh = !geometry.oh.is_multiple_of(tiling.rows);
    let ragged_ow = !geometry.ow.is_multiple_of(tiling.cols);
    let [ph, pw] = geometry.padding;
    let [sh, sw] = geometry.stride;
    let [dh, dw] = geometry.dilation;

    // Two gathered physical axes, one per spatial pair, each carrying its padding as the
    // projection's constant term. Batch and channel ride identity. The channel comes last in the
    // logical order because that is the axis the operand lines along, and the innermost logical
    // axis is the one a line covers.
    let in_spec = TileSpec::new(Projection::new(
        &[B, OH, OW, RH, RW, C],
        &[
            PhysicalAxisMap::of(B),
            PhysicalAxisMap::affine_with_offset(&[(OH, sh), (RH, dh)], -(ph as isize)),
            PhysicalAxisMap::affine_with_offset(&[(OW, sw), (RW, dw)], -(pw as isize)),
            PhysicalAxisMap::of(C),
        ],
    ))
    // The padded border reads outside the input, and the guard is what makes it read as zero.
    // Unpadded and evenly tiled, every read is in bounds by construction and the comparison is
    // paid for nothing.
    .boundaries(&[
        None,
        guard(ph > 0 || ragged_oh),
        guard(pw > 0 || ragged_ow),
        guard(ragged_c),
    ]);

    // Read in place rather than staged. Staging the filter into shared memory was measured
    // and rejected: it pays a cooperative fill and a sync per cube, which the deep blocks —
    // few output positions, many channels — never amortise, and they regressed roughly 2x
    // while only the widest spatial shapes gained.
    let w_spec = TileSpec::direct(&[RH, RW, C]).boundaries(&[None, None, guard(ragged_c)]);
    let out_spec = TileSpec::direct(&[B, OH, OW, C]).boundaries(&[
        None,
        guard(ragged_oh),
        guard(ragged_ow),
        guard(ragged_c),
    ]);

    depthwise_kernel::launch::<R>(
        client,
        space.cube_count(),
        space.cube_dim(client),
        width,
        TileArgLaunch::new(weight.into_tensor_arg(), w_spec),
        TileArgLaunch::new(input.into_tensor_arg(), in_spec),
        TileArgLaunch::new(out.into_tensor_arg(), out_spec),
        space,
        dtype,
    );

    Ok(())
}

/// The filter as `[kh, kw, c]`, contiguous.
///
/// Burn stores it `[c, kh, kw]` (its trailing `in_channels / groups` is 1 here), which is the one
/// layout this kernel cannot read: the channel has to be the innermost dim for a line to cover a
/// cell's worth of filter. Permuting the binding's existing strides re-presents that logical
/// tensor without assuming anything about its storage; `into_contiguous` is what makes the new
/// layout physical.
fn channels_innermost<R: Runtime>(
    client: &ComputeClient<R>,
    weight: TensorBinding<R>,
    geometry: &Geometry,
    dtype: ElemType,
) -> Result<TensorBinding<R>, LaunchError> {
    let (c, rh, rw) = (geometry.c, geometry.rh, geometry.rw);
    let mut permuted = weight;
    let channel_stride = permuted.strides[0];
    let row_stride = permuted.strides[1];
    let col_stride = permuted.strides[2];
    permuted.shape = Shape::from(vec![rh, rw, c]);
    // `[C, kh, kw, 1] -> [kh, kw, C]`. The omitted axis is singleton, so it contributes no
    // offset; every surviving axis must retain its actual stride for sliced/strided bindings.
    permuted.strides = Strides::new(&[row_stride, col_stride, channel_stride]);

    Ok(InputBinding::new(permuted, dtype)
        .into_contiguous(client)?
        .into_data())
}

/// The widest line the channel axis can be served in across all three operands, up to what the
/// tiling asked for.
///
/// Every gate below the request is a fact about the buffers rather than a preference: the channel
/// must be the contiguous dim, and the width must divide the channel count, since a partial line
/// has no cell to be.
fn line_width<R: Runtime>(
    client: &ComputeClient<R>,
    channels: usize,
    dtype: ElemType,
    requested: usize,
    operands: &[&TensorBinding<R>],
) -> usize {
    if !operands.iter().all(|b| b.strides.last() == Some(&1)) {
        return 1;
    }

    client
        .io_optimized_vector_sizes(dtype.size())
        .filter(|&v| {
            v <= requested
                && channels.is_multiple_of(v)
                && operands.iter().all(|b| {
                    b.strides[..b.strides.len() - 1]
                        .iter()
                        .all(|&s| s.is_multiple_of(v))
                })
        })
        .max()
        .unwrap_or(1)
}
