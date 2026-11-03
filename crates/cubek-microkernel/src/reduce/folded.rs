use crate::reduce::{ReadMap, ReduceLayout, ReduceOperation, ReduceWrites};
use cubecl::tile::{Carrier, Indexed, Monoid, TOP_K_UNROLL_BUDGET, Tile, TopK};
use cubecl::{self as cubecl, cube, prelude::*};

/// What a reduction folds a unit's lines into, `N` lanes of `Acc`: how a line read at a position
/// joins it, and how what it holds is written out. Which monoid folds it, and how the units
/// holding partials meet, is its [`Carrier`]'s.
#[cube]
pub trait Folded<Acc: Numeric, N: Size>: Carrier + CubeType<ExpandType: Assign + Copy> {
    /// `acc` with `line` folded in, the line read `line_index` lines into the row, under
    /// `monoid`.
    fn fold(
        acc: Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) -> Self;

    /// The partials of a cube's `planes` planes, each already met across its own lanes, met in
    /// shared memory: every plane's first lane writes its partial to a slot of its plane, the
    /// cube meets, every unit folds every slot, and the cube meets again so the slots may be
    /// written anew. Every unit of the cube is left with the total.
    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self;

    /// `acc` stored as partial `part` of a row its cubes split, for the cube that merges them
    /// ([`load`](Folded::load)): its lines, one for the values and for an [`Indexed`], `k` for a
    /// top-k, written that many lines times `part` into `values`, in `Acc`, and its indices at
    /// the same lines of `indices`, in `u32`, where it carries any.
    fn store(acc: Self, values: &mut Tile, indices: &mut Tile, part: usize);

    /// Partial `part`, as [`store`](Folded::store) left it.
    fn load(values: &Tile, indices: &Tile, part: usize) -> Self;

    /// What `acc` holds written as `writes` says, its lanes folded into one first where they
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

/// One lane's value and index, what an [`Indexed`] row's lanes fold into.
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
impl<Acc: Numeric, N: Size> Folded<Acc, N> for Vector<Acc, N> {
    fn fold(
        acc: Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) -> Self {
        monoid.combine::<Vector<Acc, N>>(acc, line)
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
            "Folded: {operation:?} folds the values themselves, which carry no index to write"
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
impl<Acc: Numeric, N: Size> Folded<Acc, N> for Indexed<Acc, N> {
    fn fold(
        acc: Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) -> Self {
        let read = Indexed::<Acc, N> {
            value: line,
            index: lane_indices::<N>(line_index, layout),
        };
        Indexed::<Acc, N>::combine(acc, read, monoid)
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
                // The lanes hold one row: fold them, lane by lane, as one-lane carriers.
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
impl<Acc: Numeric, N: Size, K: Size> Folded<Acc, N> for TopK<Acc, N, K> {
    fn fold(
        acc: Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) -> Self {
        acc.insert(line, lane_indices::<N>(line_index, layout), monoid)
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
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
            #[unroll(k * k <= TOP_K_UNROLL_BUDGET)]
            for slot in 0..k {
                total = total.insert(values[plane * k + slot], indices[plane * k + slot], monoid);
            }
        }
        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, indices: &mut Tile, part: usize) {
        let k = K::value();
        let mut value_lines = values.flat_mut::<Acc, N>();
        let mut index_lines = indices.flat_mut::<u32, N>();
        #[unroll(k <= TOP_K_UNROLL_BUDGET)]
        for slot in 0..k {
            value_lines.write(part * k + slot, acc.values[slot]);
            index_lines.write(part * k + slot, acc.indices[slot]);
        }
    }

    fn load(values: &Tile, indices: &Tile, part: usize) -> Self {
        let k = K::value();
        let value_lines = values.flat::<Acc, N>();
        let index_lines = indices.flat::<u32, N>();
        let mut kept_values = Array::new(k);
        let mut kept_indices = Array::new(k);
        #[unroll(k <= TOP_K_UNROLL_BUDGET)]
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
        match comptime!(layout) {
            ReduceLayout::Along => {
                // The lanes hold one row: every lane's list inserted into one one-lane list.
                let mut best = TopK::<Acc, Const<1>, K>::identity(monoid);
                #[unroll]
                for lane in 0..N::value() {
                    #[unroll(k * k <= TOP_K_UNROLL_BUDGET)]
                    for slot in 0..k {
                        best = best.insert(
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
            ReduceLayout::Across =>
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
