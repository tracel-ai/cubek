//! What a quantized store carries so its reads dequantize on their own: the scales and
//! block grid ([`QuantInfo`]) and the one site that decodes them ([`DequantAt`]).

use cubecl::{
    prelude::*,
    quant::scheme::QuantScheme,
    std::quant::view::{KnownScale, QuantizedView as DequantView},
    std::tensor::{View, layout::Coordinates},
};

use crate::*;

/// Where an operand's quantized form is decoded: the one site that turns stored values into served
/// ones. Stated at launch, once, since the quantized form ends at exactly one boundary. Which sites
/// are available is fixed by what the operand's transports can decode, never by preference, so a
/// stated value is either the one that was left (which
/// [`build`](crate::StridedTileSource::build) enforces) or a genuine fork between stage size and
/// per-read cost.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DequantAt {
    /// The load into the stage decodes; the stage holds served values, so it inflates by the
    /// served-to-stored ratio and the achievable stage depth drops with it.
    Load,
    /// The stage keeps the quantized values and their scales; the instruction's read decodes,
    /// amortized over whatever reuse the leaf has.
    Read,
}

/// Quantization a tile's store carries, so reads dequantize on their own. Holds the scale `buffer`
/// plus what walks the scales in step with the values: a per-axis `strides`, a running
/// `window_start`, and comptime `block` sizes. [`ScaleLayout`] turns those into an address ([`MemData::at`]).
/// Per-tensor is the trivial case: one scale, every stride `0`, `window_start` never moves.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct QuantInfo {
    pub(crate) buffer: Box<[f32]>,
    /// What every read below this window already holds of its scale, settled once at
    /// construction: the global level's scale read from its binding, or nothing. Never
    /// [`KnownScale::Whole`] here: a stage's scales do not exist until its fill, so a uniform
    /// window promotes to it at read time ([`dequant_view`](QuantInfo::dequant_view)).
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
    /// Where this operand's quantized form ends. Read by [`MemData::smem_like`], which is why no
    /// call site asks an operand whether it is quantized before staging it.
    #[cube(comptime)]
    pub(crate) dequant_at: DequantAt,
    /// Per-axis count of distinct scales the buffer holds, set only on a *staged* smem side-channel
    /// ([`MemData::smem_quant`]): the values stage as packed words and their scales stage compactly
    /// beside them, so the fill knows how many blocks to copy. Empty for a gmem operand, which reads
    /// the tensor's own scales in place.
    #[cube(comptime)]
    pub(crate) scale_shape: Vec<usize>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode). Always the gmem buffer: it is at
    /// most a few hundred cache-resident floats, so a stage carries it through rather than
    /// copying it ([`smem_quant`](MemData::smem_quant)).
    pub(crate) table: ComptimeOption<Box<[f32]>>,
    #[cube(comptime)]
    pub scheme: QuantScheme,
}

/// Per-axis block edges (elements per block) for a scheme. Per-tensor is one scale for the whole
/// tensor, so every axis reports `usize::MAX`: with `0` strides ([`Tile::of_dequant`]) the value
/// never addresses a real block, and it makes [`uniform_window`] report the whole window as
/// uniform, which per-tensor always is. A block scheme's edges come straight from the scheme.
pub(crate) fn block_edges(scheme: QuantScheme, rank: usize) -> Vec<usize> {
    let Some(block) = scheme.block_size() else {
        return vec![usize::MAX; rank];
    };
    block.to_dim_vec(rank).iter().map(|&b| b as usize).collect()
}

/// The [`Packing`] a quantization scheme implies: how many of its values a stored element holds
/// and what field each occupies. The one place a scheme is read for a fact about *storage*, so a
/// quantized operand and one that merely states [`TileSpec::packed`] answer every reader alike.
pub(crate) fn scheme_packing(scheme: QuantScheme) -> Packing {
    match scheme.num_quants() {
        1 => Packing::Native,
        _ => Packing::Packed {
            field: scheme.value,
        },
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
/// ([`MemData::stage_scales`] folds the global level in), so a two-level scheme stages as its
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
            .effective(self.buffer[self.window_start.fcast::<usize>()])
    }

    /// The [`DequantView`] this info's scale data resolves to for a values/scales view pair over
    /// the same coordinates: a uniform window promotes to one whole scale, read here so no read
    /// below pays for the scales view at all; any other window reads with what it already
    /// [`known`](QuantInfo::known). Shared by [`flat_transparent`](MemData::flat_transparent) and
    /// [`transparent`](MemData::transparent).
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

    /// Re-window the scales onto a tile whose absolute logical origin is `origin`. Per axis the block
    /// index is `origin / block`, dotted with the scale strides and summed into a flat start (elements
    /// everywhere, the inner axis scaled back by `vector_size`; per-tensor keeps strides `0`). Folding
    /// the window's own block index in here lets [`ScaleLayout`] add only the within-window offset,
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
            let origin_elem = origin.at(p).fmul(comptime!(w as u32).runtime());
            let block = comptime!(self.block[p] as u32).runtime();
            advances.push(origin_elem.fdiv(block).fmul(self.strides.at(p)));
        }
        QuantInfo {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            known: self.known,
            strides: self.strides.clone(),
            window_start: advances.fsum(comptime!((0..rank).collect::<Vec<_>>())),
            block: comptime!(self.block.clone()),
            extent: comptime!(extent),
            dequant_at: comptime!(self.dequant_at),
            scale_shape: comptime!(self.scale_shape.clone()),
            table: self.table.clone(),
            scheme: comptime!(self.scheme),
        }
    }
}
