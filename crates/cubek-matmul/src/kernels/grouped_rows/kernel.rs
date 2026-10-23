use cubecl::{self as cubecl, cube, prelude::*};

/// Where each tile of the padded rows reads from: one unit per group, laying
/// out its own tiles.
///
/// `ends` holds each group's cumulative end row, so group `g` owns source rows
/// `ends[g - 1]..ends[g]`. A group takes `ceil(rows / align)` tiles, in group
/// order; `table` names each tile's group and `sources` its first source row,
/// and `starts[g]` is group `g`'s first tile, `starts[ends.len()]` the tile
/// past the last group's, so group `g`'s tiles are `starts[g]..starts[g + 1]`.
/// The last group also marks every tile past the padded rows with the sentinel
/// group `ends.len()`, which a grouped product reads as no slab.
///
/// A unit sums the groups before its own, which is a loop over a handful of
/// entries rather than a scan worth a second pass.
#[cube(launch)]
pub fn group_tiles(
    ends: &Tensor<u32>,
    table: &mut Tensor<u32>,
    sources: &mut Tensor<u32>,
    starts: &mut Tensor<u32>,
    #[comptime] align: u32,
) {
    let groups = ends.len();
    let tiles = table.len();
    let group = ABSOLUTE_POS;
    if group >= groups {
        terminate!();
    }
    let mut tile = 0u32;
    let mut start = 0u32;
    for before in 0..group {
        let end = ends[before];
        tile += tiles_of(start, end, align);
        start = end;
    }
    let end = ends[group];
    let owned = tiles_of(start, end, align);
    starts[group] = tile;
    for t in 0..owned {
        let at = (tile + t) as usize;
        if at < tiles {
            table[at] = group as u32;
            sources[at] = start + t * align;
        }
    }
    if group == groups - 1 {
        starts[groups] = tile + owned;
        for at in (tile + owned) as usize..tiles {
            table[at] = groups as u32;
            sources[at] = end;
        }
    }
}

/// Tiles a group from `start` to `end` takes: none where the ends run
/// backwards, which would otherwise wrap to a count no loop finishes.
#[cube]
fn tiles_of(start: u32, end: u32, #[comptime] align: u32) -> u32 {
    let mut rows = 0u32;
    if end > start {
        rows = end - start;
    }
    rows.div_ceil(align)
}

/// The source row padded row `row` holds, and whether it holds one: a tile's
/// rows past its group's end are padding, and so is a row past the `limit`
/// rows the grouped side has, whatever the ends claim.
#[derive(CubeType)]
struct Source {
    row: usize,
    real: bool,
}

#[cube]
impl Source {
    fn new(
        row: usize,
        limit: usize,
        ends: &Tensor<u32>,
        table: &Tensor<u32>,
        sources: &Tensor<u32>,
        #[comptime] align: usize,
    ) -> Source {
        let tile = row / align;
        let group = table[tile] as usize;
        let source = sources[tile] as usize + row % align;
        let mut real = false;
        if group < ends.len() {
            real = source < ends[group] as usize && source < limit;
        }
        Source { row: source, real }
    }
}

/// `padded[row] = rows[source]` where the padded row holds a source row, zero
/// where it is padding: one unit per output vector.
/// With a route sort's `route_at_place`, grouped row `p` is read out of its token instead, row
/// `route_at_place[p] / slots` of `rows`, so the sorted copy of the tokens never exists.
/// `grouped` is how many rows are grouped.
#[cube(launch)]
pub fn pad_rows<E: Numeric, N: Size>(
    rows: &Tensor<Vector<E, N>>,
    route_at_place: &ComptimeOption<&'static Tensor<u32>>,
    padded: &mut Tensor<Vector<E, N>>,
    ends: &Tensor<u32>,
    table: &Tensor<u32>,
    sources: &Tensor<u32>,
    grouped: u32,
    slots: u32,
    #[comptime] align: usize,
    #[define(E)] _dtype: ElemType,
) {
    let at = ABSOLUTE_POS;
    if at >= padded.len() {
        terminate!();
    }
    let vectors = padded.shape(1) / padded.vector_size();
    let (row, v) = (at / vectors, at % vectors);
    let source = Source::new(row, grouped as usize, ends, table, sources, align);
    // Each row at the stride its buffer has, which may pad past the row.
    let (rows_step, padded_step) = (
        rows.stride(0) / rows.vector_size(),
        padded.stride(0) / padded.vector_size(),
    );
    let mut value = Vector::<E, N>::cast_from(0u32);
    if source.real {
        value = rows[row_read(source.row, route_at_place, slots) * rows_step + v];
    }
    padded[row * padded_step + v] = value;
}

/// The row of the source grouped row `row` reads: itself, or its route's token.
#[cube]
fn row_read(
    row: usize,
    route_at_place: &ComptimeOption<&'static Tensor<u32>>,
    slots: u32,
) -> usize {
    #[comptime]
    match route_at_place {
        ComptimeOption::Some(route_at_place) => (route_at_place[row] / slots) as usize,
        ComptimeOption::None => row,
    }
}

/// The padded row, in a layout of `align`-row tiles whose group `g` starts on tile `starts[g]`,
/// that holds grouped row `place` of group `group`.
#[cube]
fn padded_row_of(
    place: usize,
    group: usize,
    ends: &Tensor<u32>,
    starts: &Tensor<u32>,
    #[comptime] align: usize,
) -> usize {
    let mut first = 0usize;
    if group > 0 {
        first = ends[group - 1] as usize;
    }
    starts[group] as usize * align + place - first
}

/// `gated[row] = silu(gate) · up` of the grouped row padded row `row` holds, read out of `both`,
/// `[gate | up]` rows padded by another layout (`from_starts`, `from_align`), and written in
/// this one (`table`, `sources`, `align`); zero where the row is padding. The gated activation
/// moves the rows between two products' tiles in the pass it already makes. One unit per output
/// vector, in f32.
#[cube(launch)]
pub fn gate_rows<F: Float, N: Size>(
    both: &Tensor<Vector<F, N>>,
    gated: &mut Tensor<Vector<F, N>>,
    ends: &Tensor<u32>,
    table: &Tensor<u32>,
    sources: &Tensor<u32>,
    from_starts: &Tensor<u32>,
    rows: u32,
    #[comptime] align: usize,
    #[comptime] from_align: usize,
    #[define(F)] _dtype: ElemType,
) {
    let at = ABSOLUTE_POS;
    if at >= gated.len() {
        terminate!();
    }
    let vectors = gated.shape(1) / gated.vector_size();
    let (row, v) = (at / vectors, at % vectors);
    let source = Source::new(row, rows as usize, ends, table, sources, align);
    let (both_step, gated_step) = (
        both.stride(0) / both.vector_size(),
        gated.stride(0) / gated.vector_size(),
    );
    let mut out = Vector::<F, N>::cast_from(0u32);
    if source.real {
        let group = table[row / align] as usize;
        let from = padded_row_of(source.row, group, ends, from_starts, from_align);
        let gate = both[from * both_step + v];
        let up = both[from * both_step + vectors + v];
        #[unroll]
        for lane in 0..gated.vector_size() {
            let g = f32::cast_from(gate.extract(lane));
            let u = f32::cast_from(up.extract(lane));
            out.insert(lane, F::cast_from(g / (1.0 + (-g).exp()) * u));
        }
    }
    gated[row * gated_step + v] = out;
}

/// `out[t] = Σ_s weights[t, s] · grouped[padded row of route t · slots + s]`: every token's
/// routes read back out of a grouped product's padded output and summed under their weights,
/// in f32. A gather, so each output row has one writer and the sum's order is fixed. One unit per
/// output vector.
#[cube(launch)]
pub fn combine_routes<F: Float, N: Size>(
    grouped: &Tensor<Vector<F, N>>,
    routes: &Tensor<u32>,
    place_of_route: &Tensor<u32>,
    weights: &Tensor<F>,
    out: &mut Tensor<Vector<F, N>>,
    ends: &Tensor<u32>,
    starts: &Tensor<u32>,
    slots: u32,
    #[comptime] align: usize,
    #[define(F)] _dtype: ElemType,
) {
    let at = ABSOLUTE_POS;
    if at >= out.len() {
        terminate!();
    }
    let vectors = out.shape(1) / out.vector_size();
    let (token, v) = (at / vectors, at % vectors);
    let (grouped_step, out_step) = (
        grouped.stride(0) / grouped.vector_size(),
        out.stride(0) / out.vector_size(),
    );
    let slots = slots as usize;
    let mut sum = Vector::<f32, N>::cast_from(0u32);
    for slot in 0..slots {
        let route = token * slots + slot;
        let group = routes[route] as usize;
        // A route naming no group has no padded row, and adds nothing.
        if group < ends.len() {
            let row = padded_row_of(place_of_route[route] as usize, group, ends, starts, align);
            let weight =
                f32::cast_from(weights[token * weights.stride(0) + slot * weights.stride(1)]);
            let value = grouped[row * grouped_step + v];
            #[unroll]
            for lane in 0..out.vector_size() {
                let added = sum.extract(lane) + weight * f32::cast_from(value.extract(lane));
                sum.insert(lane, added);
            }
        }
    }
    let mut written = Vector::<F, N>::empty();
    #[unroll]
    for lane in 0..out.vector_size() {
        written.insert(lane, F::cast_from(sum.extract(lane)));
    }
    out[token * out_step + v] = written;
}

/// `rows[source] = padded[row]` for every padded row that holds a source row,
/// padding dropped: one unit per input vector. Each source row is written by
/// the one padded row that holds it.
#[cube(launch)]
pub fn unpad_rows<E: Numeric, N: Size>(
    padded: &Tensor<Vector<E, N>>,
    rows: &mut Tensor<Vector<E, N>>,
    ends: &Tensor<u32>,
    table: &Tensor<u32>,
    sources: &Tensor<u32>,
    #[comptime] align: usize,
    #[define(E)] _dtype: ElemType,
) {
    let at = ABSOLUTE_POS;
    if at >= padded.len() {
        terminate!();
    }
    let vectors = padded.shape(1) / padded.vector_size();
    let (row, v) = (at / vectors, at % vectors);
    let source = Source::new(row, rows.shape(0), ends, table, sources, align);
    // Each row at the stride its buffer has, which may pad past the row.
    let (rows_step, padded_step) = (
        rows.stride(0) / rows.vector_size(),
        padded.stride(0) / padded.vector_size(),
    );
    if source.real {
        rows[source.row * rows_step + v] = padded[row * padded_step + v];
    }
}
