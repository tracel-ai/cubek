//! A factor of a contraction's terms as its leaves read it: the values, and the scales it carries.
//!
//! Both leaves ask the same questions of a factor — what width its scales are read at, what the
//! level nearest the values is against them, and that level as one line source — so they are
//! asked here rather than twice. What differs between the leaves is the edges they walk and the
//! widths they serve them at, which each states for itself.

use cubecl::prelude::*;

use super::scale::{
    Apply, ContractEdges, ScaleLevel, Side, check_scales_omit_rather_than_divide, check_scales_ride,
};
use crate::instruction::registers::lines::{CombinedScales, Span};
use crate::*;

/// The width a factor's scales are read at: their own, or one where it carries none.
#[cube]
pub(crate) fn scale_width<S: Numeric>(levels: &Sequence<Tile<S>>) -> comptime_type!(usize) {
    let count = levels.len();
    if comptime!(count == 0) {
        comptime!(1usize)
    } else {
        levels.index(0).vector_size()
    }
}

/// What the level nearest the values is against them, or nothing where a factor carries none.
///
/// The coarser levels neither set the granularity nor pick the edge: each covers a tile of the
/// tiles below it and is read once per region.
#[cube]
pub(crate) fn level_of<S: Numeric>(
    levels: &Sequence<Tile<S>>,
    #[comptime] operands: Space,
    #[comptime] out: Space,
    #[comptime] acc_axes: MatrixAxes,
    #[comptime] edges: ContractEdges,
    #[comptime] side: Side,
) -> comptime_type!(Option<ScaleLevel>) {
    let count = levels.len();
    if comptime!(count == 0) {
        comptime!(None)
    } else {
        let inner = levels.index(0);
        let sw = inner.vector_size();
        let invariant = inner.invariant_over(operands);
        let projection = inner.projection();
        comptime!(check_scales_omit_rather_than_divide(&projection));
        comptime!(check_scales_ride(side, &inner.space, &out, acc_axes));
        comptime!(Some(ScaleLevel::of(
            &inner.space,
            &edges,
            side,
            &invariant,
            sw
        )))
    }
}

/// A factor's scales as one line source, or nothing where it carries none.
#[cube]
pub(crate) fn scales_of<'a, S: Numeric, SW: Size>(
    levels: &'a Sequence<Tile<S>>,
    #[comptime] level: Option<ScaleLevel>,
    mat: usize,
) -> ComptimeOption<CombinedScales<'a, S, SW>> {
    if comptime!(level.is_some()) {
        ComptimeOption::new_Some(combined_scales::<S, SW>(
            levels,
            comptime!(level.unwrap()),
            mat,
        ))
    } else {
        ComptimeOption::new_None()
    }
}

/// How a factor's lines group under its scales, or one line a run where it carries none.
pub(crate) fn span(level: Option<ScaleLevel>) -> Span {
    match level {
        Some(level) => level.span,
        None => Span::PLAIN,
    }
}

/// Every level of a scales operand as one line source.
///
/// The innermost level is read at the width its cut gives it; each coarser one is read a single
/// scale at a time and broadcast across that line. All of them are read at the same position: a
/// level resolves it to its own granularity through its own projection, which is what "one scale
/// per block" already means.
#[cube]
pub(crate) fn combined_scales<'a, ES: Numeric, S: Size>(
    scales: &'a Sequence<Tile<ES>>,
    #[comptime] level: ScaleLevel,
    mat: usize,
) -> CombinedScales<'a, ES, S> {
    let inner = scales.index(0);
    let count = scales.len();
    let origin = (0u32.runtime(), 0u32.runtime());
    let mut coarser = Vector::<ES, Const<1>>::cast_from(1);
    #[unroll]
    for k in 1..count {
        let level_above = scales.index(k);
        // Same axes at the same extents, so one `MatrixAxes` reads every level. What differs is
        // which of those axes each level's projection addresses, and that is what makes one cover
        // a tile of the other's tiles.
        comptime!(assert!(
            level_above.space.axes().collect::<Vec<_>>() == inner.space.axes().collect::<Vec<_>>()
                && (0..level_above.space.rank())
                    .all(|p| { level_above.space.extent_at(p) == inner.space.extent_at(p) }),
            "mm_scaled: a coarser scale level spans {:?} at extents {:?} where the level below it \
             spans {:?} at {:?}. Levels declare the same axes and differ by what their projections \
             address, which is what makes one cover a tile of the other's tiles",
            level_above.space.axes().collect::<Vec<_>>(),
            (0..level_above.space.rank())
                .map(|p| level_above.space.extent_at(p))
                .collect::<Vec<_>>(),
            inner.space.axes().collect::<Vec<_>>(),
            (0..inner.space.rank())
                .map(|p| inner.space.extent_at(p))
                .collect::<Vec<_>>()
        ));
        // Read once, here, which is once per region. A coarser level covers this whole region, so
        // it has no position of its own inside it; that it does is what the assert above says.
        // Read as the word it lies in, and its own scale is the word's first field.
        let above_width = level_above.vector_size();
        let size!(AW) = above_width;
        let one = Vector::<ES, Const<1>>::cast_from(
            level_above
                .matrix_packed::<AW>(comptime!(level.axes), mat)
                .read(origin)
                .extract(0usize),
        );
        match comptime!(level.apply) {
            Apply::Product => coarser *= one,
        }
    }
    CombinedScales::<ES, S>::new(
        inner.matrix_packed::<S>(comptime!(level.axes), mat),
        coarser,
    )
}
