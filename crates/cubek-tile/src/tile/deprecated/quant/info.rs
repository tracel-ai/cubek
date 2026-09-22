//! What a quantized store carries so its reads dequantize on their own: the scales and
//! block grid ([`QuantInfo`]) and the one site that decodes them ([`DequantAt`]).

use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue},
    std::quant::view::{KnownScale, QuantizedView as DequantView},
    std::tensor::{View, layout::Coordinates},
};

use crate::*;

/// Where an operand's quantized form is decoded: the one site that turns stored values into served
/// ones, stated once at launch since the quantized form ends at exactly one boundary.
///
/// Which sites are available is fixed by what the operand's transports can decode, never by
/// preference ([`build`](crate::Arg::build) enforces it); where both remain, the
/// choice is a fork between stage size and per-read cost.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DequantAt {
    /// The load into the stage decodes; the stage holds served values, so it inflates by the
    /// served-to-stored ratio and the achievable stage depth drops with it.
    Load,
    /// The stage keeps the quantized values and their scales; the instruction's read decodes,
    /// amortized over whatever reuse the leaf has.
    Read,
}

/// Quantization a tile's store carries, so reads dequantize on their own: the scale `buffer` plus
/// per-axis `strides`, a running `window_start` and comptime `block` sizes, which [`ScaleLayout`]
/// turns into an address ([`Memory::at`]). Per-tensor: one scale, every stride `0`, a fixed start.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct QuantInfo {
    pub(crate) buffer: Box<[f32]>,
    /// What every read below this window already holds of its scale, settled at construction: the
    /// global level's scale, or nothing. Never [`KnownScale::Whole`], since a stage's scales do not
    /// exist until its fill; uniform windows promote in [`dequant_view`](QuantInfo::dequant_view).
    pub(crate) known: KnownScale,
    pub(crate) strides: Coords<u32>,
    pub(crate) window_start: u32,
    #[cube(comptime)]
    pub(crate) block: Vec<usize>,
    /// Per-axis extent of the window these scales cover, in elements; [`usize::MAX`] where it is
    /// not comptime (a dynamic top-level axis). An axis whose extent fits inside a block has no
    /// distinct scales left to address, which is what [`ScaleLayout`] drops its term for.
    #[cube(comptime)]
    pub(crate) extent: Vec<usize>,
    /// Where this operand's quantized form ends. Read by [`Memory::smem_like`], which is why no
    /// call site asks an operand whether it is quantized before staging it.
    #[cube(comptime)]
    pub(crate) dequant_at: DequantAt,
    /// Per-axis count of distinct scales the buffer holds, set only on a *staged* smem side-channel
    /// ([`Memory::smem_quant`]) so the fill knows how many blocks of scales to copy beside the
    /// packed values. Empty for a gmem operand, which reads the tensor's own scales in place.
    #[cube(comptime)]
    pub(crate) scale_shape: Vec<usize>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode). Always the gmem buffer: a few
    /// hundred cache-resident floats, so a stage carries it ([`smem_quant`](Memory::smem_quant)).
    pub(crate) table: ComptimeOption<Box<[f32]>>,
    #[cube(comptime)]
    pub scheme: QuantScheme,
}

/// Per-axis block edges (elements per block) for a scheme. Per-tensor reports `usize::MAX` on
/// every axis: with `0` strides (a dequantizing operand) no real block is ever addressed, and
/// [`uniform_window`] then reports the whole window uniform, which per-tensor always is.
pub(crate) fn block_edges(scheme: QuantScheme, rank: usize) -> Vec<usize> {
    let Some(block) = scheme.block_size() else {
        return vec![usize::MAX; rank];
    };
    block.to_dim_vec(rank).iter().map(|&b| b as usize).collect()
}

/// The [`Packing`] a quantization scheme implies. The one place a scheme is read for a fact about
/// *storage*, so a quantized operand and one that merely states [`TileSpec::packed`] answer every
/// reader alike, and the one place a storage this crate does not serve (not `i8`/`u32`) is refused.
pub(crate) fn scheme_packing(scheme: QuantScheme) -> Packing {
    match scheme.store {
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S => Packing::Native,
            other => panic!("native quant storage element {other:?} is not wired (i8 only)"),
        },
        QuantStore::PackedU32(_) => Packing::Packed {
            field: scheme.value.into(),
        },
        other => panic!("quant storage {other:?} is not wired (native or packed-u32)"),
    }
}

/// Whether one scale covers a window of `extent` under `block` edges: every axis fits inside a
/// block, so there is nothing left for [`ScaleLayout`] to address and the scale can be read once
/// ([`QuantInfo::uniform_scale`]) instead of per value.
fn uniform_window(block: &[usize], extent: &[usize]) -> bool {
    (0..block.len()).all(|p| extent[p] <= block[p])
}

impl QuantInfo {
    /// See [`uniform_window`]. Both this and its expand twin exist because a `comptime!` branch
    /// typechecks as host code as well as expanded.
    pub(crate) fn uniform(&self) -> bool {
        uniform_window(&self.block, &self.extent)
    }
}

impl QuantInfoExpand {
    /// See [`uniform_window`].
    pub(crate) fn uniform(&self) -> bool {
        uniform_window(&self.block, &self.extent)
    }
}

/// Per-axis window extent in elements for a space's own level, [`usize::MAX`] where an axis is
/// dynamic. What [`QuantInfo`] carries so [`ScaleLayout`] can drop the axes that hold one scale.
pub(crate) fn window_extents(space: &Space, rank: usize) -> Vec<usize> {
    (0..rank)
        .map(|p| match space.extent_raw(space.axis_at(p)) {
            Extent::Static(e) => e,
            Extent::Dynamic => usize::MAX,
        })
        .collect()
}

/// The scheme a staged side-channel serves: its grid holds *effective* scales
/// ([`Memory::stage_scales`] folds the global level in), so a two-level scheme stages as its
/// one-level block form and reads below the stage carry no global scale.
pub(crate) fn staged_scheme(scheme: QuantScheme) -> QuantScheme {
    let Some(block) = scheme.block_scale() else {
        return scheme;
    };
    // Rebuilt rather than cleared: the levels are set additively and there is no way to drop one.
    QuantScheme::default()
        .with_value(scheme.value)
        .with_store(scheme.store)
        .with_mode(scheme.mode)
        .per_block(block.size.as_slice(), block.dtype)
}

#[cube]
impl QuantInfo {
    /// The one scale this whole window reconstructs against, global level folded in. Only
    /// meaningful where [`uniform`](QuantInfoExpand::uniform) holds; one load for the whole tile.
    pub(crate) fn uniform_scale(&self) -> f32 {
        self.known
            .effective(self.buffer[self.window_start.retyped::<usize>()])
    }

    /// The [`DequantView`] over a values/scales view pair on the same coordinates. Shared by
    /// [`flat_transparent`](Memory::flat_transparent) and [`transparent`](Memory::transparent).
    ///
    /// A uniform window promotes to one whole scale read here, so no read below pays for the
    /// scales view; any other window reads with what it already [`known`](QuantInfo::known).
    pub(crate) fn dequant_view<
        'a,
        I: Numeric,
        WP: Size,
        T: Numeric,
        W: Size,
        C: Coordinates + 'static,
    >(
        &self,
        values: View<'a, Vector<I, WP>, C>,
        scales: View<'a, f32, C>,
    ) -> DequantView<'a, I, WP, f32, T, W, C> {
        let known = if comptime!(self.uniform()) {
            KnownScale::new_Whole(self.uniform_scale())
        } else {
            self.known
        };
        DequantView::<I, WP, f32, T, W, C>::new_with_known_scale(
            values,
            scales,
            known,
            self.table.clone(),
            comptime!(self.scheme),
        )
    }

    /// Re-window the scales onto a tile whose absolute logical origin is `origin`: per axis the
    /// block index `origin / block` (in elements, the inner axis scaled by `vector_size`) is dotted
    /// with the scale strides into a flat start; per-tensor keeps strides `0`.
    ///
    /// Folding the block index in here lets [`ScaleLayout`] add only the within-window offset,
    /// sound because a window never straddles a block (`validate_scheme` enforces it).
    pub(crate) fn window(
        &self,
        origin: &Coords<u32>,
        #[comptime] rank: usize,
        #[comptime] vector_size: usize,
        #[comptime] extent: Vec<usize>,
    ) -> QuantInfo {
        let last = comptime!(rank - 1);
        let mut advances = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let w = comptime!(if p == last { vector_size } else { 1usize });
            let origin_elem = origin.at(p).times(comptime!(w as u32).runtime());
            let block = comptime!(self.block[p] as u32).runtime();
            advances.push(origin_elem.divided_by(block).times(self.strides.at(p)));
        }
        QuantInfo {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            known: self.known,
            strides: self.strides.clone(),
            window_start: advances.sum(comptime!((0..rank).collect::<Vec<_>>())),
            block: comptime!(self.block.clone()),
            extent: comptime!(extent),
            dequant_at: comptime!(self.dequant_at),
            scale_shape: comptime!(self.scale_shape.clone()),
            table: self.table.clone(),
            scheme: comptime!(self.scheme),
        }
    }
}
