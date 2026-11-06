use crate::reduce::{ReadMap, ReduceLayout, ReduceOperation, ReduceWrites};
use cubecl::tile::{Carrier, Indexed, Monoid, TOP_K_UNROLL_BUDGET, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// What a reduction accumulates a unit's lines into, `N` lanes of `Acc`: how a line read at a position
/// joins it, and how what it holds is written out. Which monoid combines it, and how the units
/// holding partials meet, is its [`Carrier`]'s.
#[cube]
pub trait ReduceAccumulator<Acc: Numeric, N: Size>:
    Carrier + CubeType<ExpandType: Assign + Copy>
{
    /// `line` accumulated into `acc`, the line read `line_index` lines into the row, under
    /// `monoid`. In place, since it runs once a line read: a carrier returned anew is copied back
    /// slot by slot, which for a top-k's `K` slots costs more than the accumulation.
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    );

    /// The partials of a cube's `planes` planes, each already met across its own lanes, met in
    /// shared memory: every plane's first lane writes its partial to a slot of its plane, the
    /// cube meets, every unit combines every slot, and the cube meets again so the slots may be
    /// written anew. Every unit of the cube is left with the total.
    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self;

    /// `acc` stored as partial `part` of a row its cubes split, for the cube that merges them
    /// ([`load`](ReduceAccumulator::load)): its lines, one for the values and for an [`Indexed`], `k` for a
    /// top-k, written that many lines times `part` into `values`, in `Acc`, and its indices at
    /// the same lines of `indices`, in `u32`, where it carries any.
    fn store(acc: Self, values: &mut Tile, indices: &mut Tile, part: usize);

    /// Partial `part`, as [`store`](ReduceAccumulator::store) left it.
    fn load(values: &Tile, indices: &Tile, part: usize) -> Self;

    /// What `acc` holds written as `writes` says, its lanes reduced to one first where they
    /// hold one row ([`ReduceLayout::Along`]): to `out`'s first cell along the row, or its first
    /// line across rows (its first `k` of either, for a top-k). With [`ReduceWrites::Both`] the
    /// indices go to `indices` at the same places, in `Idx`; `indices` is written only then.
    /// `count` is the number of values reduced, which a mean divides by.
    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        indices: &mut Tile,
        count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    );
}

/// One lane's value and index, what an [`Indexed`] row's lanes reduce to.
type OneIndexed<Acc> = Indexed<Acc, Const<1>>;

/// `line` as `map` reads it.
#[cube]
pub fn read_mapped<Acc: Numeric, N: Size>(
    line: Vector<Acc, N>,
    #[comptime] map: ReadMap,
) -> Vector<Acc, N> {
    match comptime!(map) {
        ReadMap::AsRead => line,
        ReadMap::Abs => line.abs(),
        ReadMap::NonZero => select_many(
            line.not_equal(&Vector::new(Acc::from_int(0))),
            Vector::new(Acc::from_int(1)),
            Vector::new(Acc::from_int(0)),
        ),
    }
}

/// The index of each lane of the line read `line_index` lines into the row: along the row,
/// consecutive values; across rows, the line's own position, each lane in its own row.
#[cube]
fn lane_indices<N: Size>(line_index: usize, #[comptime] layout: ReduceLayout) -> Vector<u32, N> {
    match comptime!(layout) {
        ReduceLayout::Along => {
            let width = N::value();
            let mut indices = Vector::<u32, N>::new((line_index * width) as u32);
            #[unroll]
            for lane in 0..width {
                indices.insert(lane, (line_index * width + lane) as u32);
            }
            indices
        }
        ReduceLayout::Across => Vector::new(line_index as u32),
    }
}

/// The values themselves, pointwise on the line.
#[cube]
impl<Acc: Numeric, N: Size> ReduceAccumulator<Acc, N> for Vector<Acc, N> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        *acc = monoid.combine::<Vector<Acc, N>>(*acc, line);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let mut slots = Shared::<[Vector<Acc, N>]>::new_slice(planes);
        if UNIT_POS_PLANE == 0 {
            slots[PLANE_POS as usize] = acc;
        }
        sync_cube();
        let mut total = Vector::<Acc, N>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            total = monoid.combine::<Vector<Acc, N>>(total, slots[plane]);
        }
        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, _indices: &mut Tile, part: usize) {
        let mut lines = values.flat_mut::<Acc, N>();
        lines.write(part, acc);
    }

    fn load(values: &Tile, _indices: &Tile, part: usize) -> Self {
        values.flat::<Acc, N>().read(part)
    }

    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        _indices: &mut Tile,
        count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        comptime!(assert!(
            writes == ReduceWrites::Values,
            "ReduceAccumulator: {operation:?} accumulates the values themselves, which carry no index to write"
        ));
        let monoid = comptime!(operation.monoid());
        match comptime!(layout) {
            ReduceLayout::Along => {
                let mut total = Monoid::reduce::<Acc, N>(acc, N::value(), monoid);
                if comptime!(operation.divides_by_count()) {
                    total = Acc::cast_from(f32::cast_from(total) / count as f32);
                }
                let mut cells = out.flat_mut::<Out, Const<1>>();
                cells.write(0, Vector::cast_from(total));
            }
            ReduceLayout::Across => {
                let mut total = acc;
                if comptime!(operation.divides_by_count()) {
                    total = Vector::cast_from(
                        Vector::<f32, N>::cast_from(total) / Vector::new(count as f32),
                    );
                }
                let mut lines = out.flat_mut::<Out, N>();
                lines.write(0, Vector::cast_from(total));
            }
        }
    }
}

/// A value with the index it was read at: the argmax and the argmin.
#[cube]
impl<Acc: Numeric, N: Size> ReduceAccumulator<Acc, N> for Indexed<Acc, N> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        let read = Indexed::<Acc, N> {
            value: line,
            index: lane_indices::<N>(line_index, layout),
        };
        *acc = Indexed::<Acc, N>::combine(*acc, read, monoid);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(planes);
        let mut indices = Shared::<[Vector<u32, N>]>::new_slice(planes);
        if UNIT_POS_PLANE == 0 {
            values[PLANE_POS as usize] = acc.value;
            indices[PLANE_POS as usize] = acc.index;
        }
        sync_cube();
        let mut total = Indexed::<Acc, N>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            let slot = Indexed::<Acc, N> {
                value: values[plane],
                index: indices[plane],
            };
            total = Indexed::<Acc, N>::combine(total, slot, monoid);
        }
        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, indices: &mut Tile, part: usize) {
        let mut value_lines = values.flat_mut::<Acc, N>();
        value_lines.write(part, acc.value);
        let mut index_lines = indices.flat_mut::<u32, N>();
        index_lines.write(part, acc.index);
    }

    fn load(values: &Tile, indices: &Tile, part: usize) -> Self {
        Indexed::<Acc, N> {
            value: values.flat::<Acc, N>().read(part),
            index: indices.flat::<u32, N>().read(part),
        }
    }

    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        indices: &mut Tile,
        _count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        let monoid = comptime!(operation.monoid());
        match comptime!(layout) {
            ReduceLayout::Along => {
                // The lanes hold one row: reduce them, lane by lane, as one-lane carriers.
                let mut best = OneIndexed::<Acc>::identity(monoid);
                #[unroll]
                for lane in 0..N::value() {
                    let one = OneIndexed::<Acc> {
                        value: Vector::new(acc.value.extract(lane)),
                        index: Vector::new(acc.index.extract(lane)),
                    };
                    best = OneIndexed::<Acc>::combine(best, one, monoid);
                }
                write_line::<Out, Idx, Acc, Const<1>>(
                    out, indices, 0, best.value, best.index, writes,
                );
            }
            ReduceLayout::Across => {
                write_line::<Out, Idx, Acc, N>(out, indices, 0, acc.value, acc.index, writes);
            }
        }
    }
}

/// Line `at` of what an owner writes, `value` and its `index`, written as `writes` says: one of
/// them to `out`, or the value to `out` and the index to `indices`, which nothing else writes.
#[cube]
fn write_line<Out: Numeric, Idx: Numeric, Acc: Numeric, W: Size>(
    out: &mut Tile,
    indices: &mut Tile,
    at: usize,
    value: Vector<Acc, W>,
    index: Vector<u32, W>,
    #[comptime] writes: ReduceWrites,
) {
    let mut lines = out.flat_mut::<Out, W>();
    match comptime!(writes) {
        ReduceWrites::Values => lines.write(at, Vector::cast_from(value)),
        ReduceWrites::Indices => lines.write(at, Vector::cast_from(index)),
        ReduceWrites::Both => {
            lines.write(at, Vector::cast_from(value));
            let mut index_lines = indices.flat_mut::<Idx, W>();
            index_lines.write(at, Vector::cast_from(index));
        }
    }
}

/// The `K` best values with their indices: the top-k.
#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopK<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, lane_indices::<N>(line_index, layout), monoid);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(comptime!(planes * k));
        let mut indices = Shared::<[Vector<u32, N>]>::new_slice(comptime!(planes * k));
        if UNIT_POS_PLANE == 0 {
            #[unroll]
            for slot in 0..k {
                values[PLANE_POS as usize * k + slot] = acc.values[slot];
                indices[PLANE_POS as usize * k + slot] = acc.indices[slot];
            }
        }
        sync_cube();
        let mut total = TopK::<Acc, N, K>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            #[unroll(unrolled)]
            for slot in 0..k {
                total.insert_in_full_order(
                    values[plane * k + slot],
                    indices[plane * k + slot],
                    monoid,
                );
            }
        }
        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, indices: &mut Tile, part: usize) {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);
        let mut value_lines = values.flat_mut::<Acc, N>();
        let mut index_lines = indices.flat_mut::<u32, N>();
        #[unroll(unrolled)]
        for slot in 0..k {
            value_lines.write(part * k + slot, acc.values[slot]);
            index_lines.write(part * k + slot, acc.indices[slot]);
        }
    }

    fn load(values: &Tile, indices: &Tile, part: usize) -> Self {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);
        let value_lines = values.flat::<Acc, N>();
        let index_lines = indices.flat::<u32, N>();
        let mut kept_values = Array::new(k);
        let mut kept_indices = Array::new(k);
        #[unroll(unrolled)]
        for slot in 0..k {
            kept_values[slot] = value_lines.read(part * k + slot);
            kept_indices[slot] = index_lines.read(part * k + slot);
        }
        TopK::<Acc, N, K>::new(kept_values, kept_indices)
    }

    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        indices: &mut Tile,
        _count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        let monoid = comptime!(operation.monoid());
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        // The lanes hold one row: every lane's list inserted into one one-lane list. One lane
        // already holds the row's list, best first, and writes it as it is.
        let lanes = N::value();
        let lanes_share_a_row = comptime!(matches!(layout, ReduceLayout::Along) && lanes > 1);
        match comptime!(lanes_share_a_row) {
            true => {
                let mut best = TopK::<Acc, Const<1>, K>::identity(monoid);
                #[unroll]
                for lane in 0..N::value() {
                    #[unroll(unrolled)]
                    for slot in 0..k {
                        best.insert_in_full_order(
                            Vector::new(acc.values[slot].extract(lane)),
                            Vector::new(acc.indices[slot].extract(lane)),
                            monoid,
                        );
                    }
                }
                #[unroll]
                for slot in 0..k {
                    write_line::<Out, Idx, Acc, Const<1>>(
                        out,
                        indices,
                        slot,
                        best.values[slot],
                        best.indices[slot],
                        writes,
                    );
                }
            }
            false =>
            {
                #[unroll]
                for slot in 0..k {
                    write_line::<Out, Idx, Acc, N>(
                        out,
                        indices,
                        slot,
                        acc.values[slot],
                        acc.indices[slot],
                        writes,
                    );
                }
            }
        }
    }
}

/// The `K` best values without their indices: a top-k that writes only its values.
#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopValues<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, monoid);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(comptime!(planes * k));
        if UNIT_POS_PLANE == 0 {
            #[unroll]
            for slot in 0..k {
                values[PLANE_POS as usize * k + slot] = acc.values[slot];
            }
        }
        sync_cube();
        let mut total = TopValues::<Acc, N, K>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            #[unroll(unrolled)]
            for slot in 0..k {
                total.insert(values[plane * k + slot], monoid);
            }
        }
        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, _indices: &mut Tile, part: usize) {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);
        let mut value_lines = values.flat_mut::<Acc, N>();
        #[unroll(unrolled)]
        for slot in 0..k {
            value_lines.write(part * k + slot, acc.values[slot]);
        }
    }

    fn load(values: &Tile, _indices: &Tile, part: usize) -> Self {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);
        let value_lines = values.flat::<Acc, N>();
        let mut kept_values = Array::new(k);
        #[unroll(unrolled)]
        for slot in 0..k {
            kept_values[slot] = value_lines.read(part * k + slot);
        }
        TopValues::<Acc, N, K>::new(kept_values)
    }

    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        _indices: &mut Tile,
        _count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] _writes: ReduceWrites,
    ) {
        let monoid = comptime!(operation.monoid());
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        // The lanes hold one row: every lane's list inserted into one one-lane list. One lane
        // already holds the row's list, best first, and writes it as it is.
        let lanes = N::value();
        let lanes_share_a_row = comptime!(matches!(layout, ReduceLayout::Along) && lanes > 1);
        match comptime!(lanes_share_a_row) {
            true => {
                let mut best = TopValues::<Acc, Const<1>, K>::identity(monoid);
                #[unroll]
                for lane in 0..N::value() {
                    #[unroll(unrolled)]
                    for slot in 0..k {
                        best.insert(Vector::new(acc.values[slot].extract(lane)), monoid);
                    }
                }
                let mut lines = out.flat_mut::<Out, Const<1>>();
                #[unroll]
                for slot in 0..k {
                    lines.write(slot, Vector::cast_from(best.values[slot]));
                }
            }
            false => {
                let mut lines = out.flat_mut::<Out, N>();
                #[unroll]
                for slot in 0..k {
                    lines.write(slot, Vector::cast_from(acc.values[slot]));
                }
            }
        }
    }
}
