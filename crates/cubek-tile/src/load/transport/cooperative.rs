//! The straight-line transport: the destination filled in its own physical order, whole lines,
//! decoding the source once per line, the cube's units taking the lines cyclically between them.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use super::padded::{physical_pos, read_stage_line, widened_shape};
use crate::*;

#[cube]
impl<T: Numeric> Memory<T> {
    /// The straight-line half of [`fill_from`](Memory::fill_from): the destination filled in its
    /// own physical order, whole `Vector<I2, WP2>` lines, decoding the source once per line. `I2` /
    /// `WP2`: the *storage* element and width, `T` for a plain copy, `u32`/`i8` for a quant stage.
    ///
    /// Both sides are physical boxes of the same rank here, so the fill copies and never gathers;
    /// a gathered pair differs only by the compaction's step. The [`Window`] sits below either way,
    /// so a cell past the source's bound masks to zero once, at fill, not at every read.
    pub(crate) fn fill_straight<I2: Numeric, WP2: Size>(
        &mut self,
        src: &Memory<T>,
        #[comptime] space: Space,
    ) {
        // A gathered stage owns mutable map registers alongside its bytes. Store the source
        // window's coefficients and phase into those registers so bytes and interpretation are
        // one slot value. Direct stages carry no runtime map state.
        if comptime!(self.projection.is_rational() || self.projection.has_dynamic_scales()) {
            self.map.store_from(&src.map);
        }
        let check = comptime!(src.access.overhang.masks());
        let sw = comptime!(src.store.vector_size);
        let w = comptime!(self.store.vector_size);
        let compaction = comptime!(stage_compaction(
            &src.projection,
            &self.projection,
            w,
            &space
        ));
        // Empty exactly when the window has no holes to skip, so the fill reads the source box
        // straight through and this layer is never built.
        let steps = comptime!(match &compaction {
            Some(c) if !c.is_dense() => c.steps().to_vec(),
            _ => Vec::new(),
        });
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        let total = shape
            .product(comptime!((0..plen).collect::<Vec<_>>()))
            .cast::<usize>();
        let projection = comptime!(self.layout.projection.clone());
        let rows = comptime!(self.layout.rows);
        let strides = self.layout.physical_strides.clone();
        // Asked whatever the widths: an equal-width fill reads nothing off the extent, but owes
        // the same agreement between the two boxes.
        let extent = comptime!(fill_extent(&space, sw, w, check));
        let src_rank = comptime!(src.projection.physical_rank());
        let padding = comptime!((sw != w).then(|| {
            // `source_component` swaps the innermost entry of a destination coordinate to address the
            // source, which only lands on a source cell when the two boxes have the same rank. A
            // storage-tiled stage splits each axis into a grid and a block digit and does not.
            assert!(
                src_rank == plen,
                "Memory::fill_straight: a padded stage is a rank-{plen} box filled from a \
                 rank-{src_rank} source, so a destination coordinate does not address it"
            );
            Padding {
                width: w,
                extent,
                rank: src_rank,
            }
        }));
        // A comptime worker count emits the tasks straight-line: a rolled loop's runtime `CUBE_DIM`
        // stride blocks unrolling, and on Metal's in-order pipe each store then stalls the next
        // read. Only a spilling last task needs a guard; unknown or tiny cubes stay rolled.
        //
        // `constant()` bridges the folded total back to host data; a whole smem stage's shape is
        // static, so it always folds.
        let units = comptime!(self.access.units);
        let total_c = total.constant();
        // The other half of the fill's contract: the mappings agree ([`stage_compaction`]), and so
        // do the sizes. A gathered destination is always an smem stage, so its line count folds and
        // has to be exactly the compacted window's.
        let cells = comptime!(compaction.as_ref().map(|c| c.cells(w)));
        comptime!(assert!(
            match cells {
                Some(n) => matches!(total_c, Some(t) if t as usize == n),
                None => true,
            },
            "Memory::fill_straight: a gathered source fills a destination of {total_c:?} lines, \
             but its compacted window is {cells:?}"
        ));
        let straight =
            comptime!(matches!(total_c, Some(t) if units > 0 && (t as usize).div_ceil(units) <= 8));
        let d = self.lines_storage_mut::<I2, WP2>();
        if comptime!(sw == w) {
            let s = if comptime!(steps.is_empty()) {
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked)),
                    check,
                )
            } else {
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked))
                        .view(CompactionStep::new(shape.clone(), comptime!(steps))),
                    check,
                )
            };
            fill_lines::<I2, WP2, WP2>(
                d, &s, projection, rows, &shape, &strides, total, total_c, units, straight, padding,
            );
        } else {
            let s = if comptime!(steps.is_empty()) {
                Masked::new(
                    src.window_view_storage::<I2, Const<1>>(comptime!(Guard::Checked)),
                    check,
                )
            } else {
                Masked::new(
                    src.window_view_storage::<I2, Const<1>>(comptime!(Guard::Checked))
                        .view(CompactionStep::new(
                            widened_shape(&shape, comptime!(plen), comptime!(w)),
                            comptime!(steps),
                        )),
                    check,
                )
            };
            fill_lines::<I2, WP2, Const<1>>(
                d, &s, projection, rows, &shape, &strides, total, total_c, units, straight, padding,
            );
        }
    }
}

/// The innermost extent of `space` in cells, with the two widths a fill pairs checked against it.
///
/// The fill reads whole `sw`-wide source lines, so the innermost extent has to be a whole number
/// of them; only the *destination* may hold a partial `w`-wide line (a padded stage, spare units
/// zero). Else the stage rounds its line count up ([`storage_extents`]) where the source truncates.
///
/// `None` for a `Dynamic` extent: nothing can be said at comptime, so a padded stage over one
/// leans on `check` to zero its spare units instead.
pub(crate) fn fill_extent(space: &Space, sw: usize, w: usize, check: bool) -> Option<usize> {
    match space.extent_raw(space.axis_at(space.rank() - 1)) {
        Extent::Static(e) => {
            assert!(
                e.is_multiple_of(sw),
                "Memory: the innermost extent {e} is not a whole number of the source's \
                 {sw}-wide lines, so the stage holds cells the source cannot hand it"
            );
            Some(e)
        }
        Extent::Dynamic => {
            assert!(
                sw == w || check,
                "Memory: a padded stage over a Dynamic innermost extent cannot know at comptime \
                 which units are padding, so its source must be bounds-checked for them to read \
                 as zero"
            );
            None
        }
    }
}

/// Schedule cooperative cyclic writing of destination stage lines across cube units.
///
/// Dispatches line reads via [`read_stage_line`], taking an unrolled loop when the task count
/// is small and static (`straight == true`) or a dynamic `CUBE_DIM`-strided while loop otherwise.
#[cube]
pub(crate) fn fill_lines<I2: Numeric, WP2: Size, SW: Size>(
    d: &mut [Vector<I2, WP2>],
    s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
    #[comptime] projection: Projection,
    #[comptime] rows: RowPlacement,
    shape: &Coords<u32>,
    strides: &Coords<u32>,
    total: usize,
    #[comptime] total_c: Option<u64>,
    #[comptime] units: usize,
    #[comptime] straight: bool,
    #[comptime] padding: Option<Padding>,
) {
    if comptime!(straight) {
        let tasks = comptime!((total_c.unwrap() as usize).div_ceil(units));
        #[unroll]
        for t in 0..tasks {
            let i = UNIT_POS as usize + comptime!(t * units);
            if comptime!((t + 1) * units > total_c.unwrap() as usize) {
                if i < total {
                    d[stage_offset(rows, i, shape, strides)] = read_stage_line::<I2, WP2, SW>(
                        s,
                        &physical_pos(comptime!(projection.clone()), rows, i, shape),
                        comptime!(padding),
                    );
                }
            } else {
                d[stage_offset(rows, i, shape, strides)] = read_stage_line::<I2, WP2, SW>(
                    s,
                    &physical_pos(comptime!(projection.clone()), rows, i, shape),
                    comptime!(padding),
                );
            }
        }
    } else {
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            d[stage_offset(rows, i, shape, strides)] = read_stage_line::<I2, WP2, SW>(
                s,
                &physical_pos(comptime!(projection.clone()), rows, i, shape),
                comptime!(padding),
            );
            i += workers;
        }
    }
}
