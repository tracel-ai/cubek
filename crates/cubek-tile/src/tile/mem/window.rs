//! Addressing the bytes: [`GmemLayout`] dots the physical strides, [`Window`] boxes the part of
//! the buffer a tile is looking at, and [`SourceWindow`] is the producer-side box a fused read
//! proves its coordinates against.

use cubecl::zspace::SmallVec;
use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// In-kernel twin of cubecl's `TiledViewLayout`, which has no in-kernel constructor: splits each
/// coordinate into the digits its [`projection`](Projection) spreads over the physical axes, then
/// dots the physical strides. Folding arithmetic, so a static store (smem) splits and dots by
/// constants, and an untiled projection (one physical axis per logical one, so every digit is the
/// whole coordinate) reduces to the plain strided dot. `Coordinates` are already physical (any
/// gather is resolved a layer up, by [`AxisProjection`]), so `projection` here is always
/// [`Projection::of_tiling`]'s synthetic per-position map, not the operand's own.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct GmemLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) projection: Projection,
}

#[cube]
impl Layout for GmemLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        // Per-physical-axis terms, summed below (chained, so a static store's dot folds).
        let mut terms = Sequence::<u32>::new();
        let rank = comptime!(self.projection.physical_rank());
        #[unroll]
        for pa in 0..rank {
            let map = comptime!(self.projection.physical_axis(pa).clone());
            let picks = comptime!((0..map.terms().len()).collect::<Vec<_>>());
            // Almost always one term (only a gather layers several onto one physical axis, and
            // `GmemLayout`'s own map never does); summed the same way regardless.
            let mut parts = Sequence::<u32>::new();
            #[unroll]
            for t in 0..comptime!(map.terms().len()) {
                let term = comptime!(map.terms()[t]);
                let p = comptime!(self.projection.position(term.axis));
                let (finer, modulo) = comptime!(self.projection.digit(pa, term.axis));
                // Strip the finer digits, then take this one. The outermost fragment of an axis
                // (and any untiled axis) has no radix and keeps the full quotient.
                let quot = pos[p].fdiv(self.physical_shape.fproduct(comptime!(finer.to_vec())));
                let digit = match comptime!(modulo) {
                    Some(m) => quot.frem(self.physical_shape.at(m)),
                    None => quot,
                };
                parts.push(digit.fmul(comptime!(term.scale.get() as u32).runtime()));
            }
            terms.push(parts.fsum(picks).fmul(self.physical_strides.at(pa)));
        }
        terms
            .fsum(comptime!((0..rank).collect::<Vec<_>>()))
            .fcast::<usize>()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        logical_extent(comptime!(self.projection.clone()), &self.physical_shape).to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;

        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }

        valid
    }
}

/// The layout [`Tile::at`] applies: shift every axis to `origin` and crop it to
/// `extent`. Same rank as the source; the rank-reducing 2-D slice is
/// [`TileMatrix`](super::TileMatrix).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Window {
    pub(crate) origin: Coords<i32>,
    pub(crate) extent: Coords<u32>,
    /// Absolute logical extent (the valid region). `shape()` stays `extent` (the tile
    /// cell, so loops cover the whole padded tile), but `is_in_bounds` clips against
    /// `bound` so a checked read/write zeroes / skips the overhang.
    pub(crate) bound: Coords<u32>,
    /// Whether the origin can be negative.
    #[cube(comptime)]
    pub(crate) signed: bool,
    /// Per-coordinate-axis boundary handling, same rank as `bound`. `None` means that axis is in
    /// bounds by construction; an empty list makes every axis `None`.
    #[cube(comptime)]
    pub(crate) boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
}

#[cube]
impl Window {
    pub fn new(
        origin: Coords<i32>,
        extent: Coords<u32>,
        bound: Coords<u32>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
    ) -> Self {
        // Both walks index `origin`, `pos` and `boundaries` by one counter, so a rank slip there
        // would silently apply one axis's mode to another rather than fail. `bound` is left out:
        // a sub-window inherits its parent's, which on a stage is the buffer's own rank (a tiled
        // stage's fragments) rather than the coordinate rank the origin is at.
        let origin_rank = origin.len();
        let extent_rank = extent.len();
        comptime!(assert!(
            extent_rank == origin_rank
                && (boundaries.is_empty() || boundaries.len() == origin_rank),
            "Window: origin ({origin_rank}), extent ({extent_rank}) and boundaries ({}) index the \
             same axes and must agree in rank",
            boundaries.len()
        ));
        Window {
            origin,
            extent,
            bound,
            signed,
            boundaries,
        }
    }

    /// Whether `pos` is valid on the selected physical axes. This is the factor-local form of
    /// [`Layout::is_in_bounds`]: separable normalization must mask the source axes moved by one
    /// tap without letting another factor's tap affect that decision.
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    pub(crate) fn axes_in_bounds(&self, pos: &CoordsDyn, #[comptime] axes: Vec<usize>) -> bool {
        let mut valid = true;
        #[unroll]
        for a in 0..comptime!(axes.len()) {
            let i = comptime!(axes[a]);
            if comptime!(self.boundaries.get(i).copied().flatten() == Some(Boundary::Zero)) {
                valid = valid && self.axis_in_bounds(pos[i], i);
            }
        }
        valid
    }

    /// The scalar physical-axis check behind [`axes_in_bounds`](Self::axes_in_bounds).
    pub(crate) fn axis_in_bounds(&self, pos: u32, #[comptime] axis: usize) -> bool {
        if comptime!(self.boundaries.get(axis).copied().flatten() == Some(Boundary::Zero)) {
            let abs = self.origin.at(axis).fadd(pos.fcast::<i32>());
            if comptime!(self.signed) {
                abs >= 0i32 && abs.fcast::<u32>() < self.bound.at(axis)
            } else {
                abs.fcast::<u32>() < self.bound.at(axis)
            }
        } else {
            true.runtime()
        }
    }
}

/// Where a gathered stage sits inside the buffer it was filled from.
///
/// A stage is addressed by [`Compaction`](crate::Compaction)'s projection, which keeps the source
/// map's terms and drops its offset, so a staged coordinate `c` lands on `origin + c * step` in
/// the source. The fill wrote the boundary's value wherever that landed outside, and the staged
/// window cannot say which cells those were; this is what lets a reader put the question to the
/// source rectangle instead.
///
/// Invariant under [`at`](MemData::at): a region step moves the staged window and the source
/// window by the same physical delta, so only the staged origin has to move and this stays as it
/// was filled.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct SourceWindow {
    /// The source window's origin, as [`fill_from`](MemData::fill_from) found it.
    pub(crate) origin: Coords<i32>,
    /// The source buffer's logical extent, which is what a tap is in bounds against.
    pub(crate) bound: Coords<u32>,
    /// What a stage coordinate is multiplied by to land on the source, per physical axis.
    #[cube(comptime)]
    pub(crate) steps: SmallVec<[usize; MAX_AXES]>,
    /// Whether the source origin can be negative.
    #[cube(comptime)]
    pub(crate) signed: bool,
    /// The source's per-axis boundary handling. Same meaning as [`Window::boundaries`], read off
    /// the operand the stage was filled from rather than the stage's own (empty) list.
    #[cube(comptime)]
    pub(crate) boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
}

#[cube]
impl SourceWindow {
    /// Whether one physical coordinate of the staged window lands inside the source.
    ///
    /// `stage_origin` is the staged window's own origin on this axis and `pos` the offset within
    /// it, so `stage_origin + pos` is the stage coordinate the caller is asking about. An axis the
    /// source did not pad is in bounds by construction, exactly as it is on a [`Window`].
    pub(crate) fn axis_in_bounds(
        &self,
        stage_origin: i32,
        pos: u32,
        #[comptime] axis: usize,
    ) -> bool {
        if comptime!(self.boundaries.get(axis).copied().flatten() == Some(Boundary::Zero)) {
            let step = comptime!(self.steps.get(axis).copied().unwrap_or(1) as i32);
            let cell = (stage_origin + pos.fcast::<i32>()) * step;
            let abs = self.origin.at(axis) + cell;
            if comptime!(self.signed) {
                abs >= 0i32 && abs.fcast::<u32>() < self.bound.at(axis)
            } else {
                abs.fcast::<u32>() < self.bound.at(axis)
            }
        } else {
            true.runtime()
        }
    }
}

#[cube]
impl Window {
    /// This window under `guard`: [`Guard::Proved`] drops the boundary machinery (no clamp for
    /// an origin that can go negative, and no per-axis [`Boundary`] mode), which is work whose
    /// answer the reader already knows and the window would otherwise pay for once per access.
    ///
    /// One path either way, differing only in comptime fields. A branch here would be a runtime
    /// select over the window's *runtime* halves as well, which is both slower and, on the
    /// accelerated leaves that share this constructor, wrong.
    pub(crate) fn with_guard(self, #[comptime] guard: Guard) -> Window {
        Window {
            origin: self.origin,
            extent: self.extent,
            bound: self.bound,
            signed: comptime!(guard.checks() && self.signed),
            boundaries: comptime!(match guard {
                Guard::Checked => self.boundaries.clone(),
                Guard::Proved => SmallVec::new(),
            }),
        }
    }
}

#[cube]
impl Layout for Window {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for i in 0..self.origin.len() {
            let abs = self.origin.at(i).fadd(pos[i].fcast::<i32>());
            // Clamp negative coordinates to 0 before bounds masking. Branchless: this runs per
            // tap of every gathered read, where a diamond would cost more than the cast it skips.
            let shifted = if comptime!(self.signed) {
                select(abs >= 0i32, abs.fcast::<u32>(), 0u32)
            } else {
                abs.fcast::<u32>()
            };
            // Under `Clamp`, fold this coordinate onto its axis's edge cell rather than
            // leaving it for the mask.
            let shifted = match comptime!(self.boundaries.get(i).copied().flatten()) {
                Some(Boundary::Clamp) => {
                    let bound_i = self.bound.at(i);
                    let edge = select(shifted >= bound_i, bound_i.fsub(1u32), shifted);
                    // A zero-extent axis has no edge cell to fold onto, and the `bound - 1` above
                    // wrapped into a wild line index; both arms evaluate, so it is discarded here
                    // rather than skipped, and the axis folds to `0` like an underflow instead.
                    select(bound_i == 0u32, 0u32, edge)
                }
                None | Some(Boundary::Zero) => shifted,
            };
            out.push(shifted);
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.axes_in_bounds(
            &pos,
            comptime!((0..self.boundaries.len()).collect::<Vec<_>>()),
        )
    }
}
