//! Lowering `c.reduce_axis(input, inst)`: at a final tile, the leaf instruction; while levels remain,
//! walk this level under its [`Schedule`].

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use super::kind::{ReduceLeafKind, identity};
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.reduce_axis(input, inst)`: reduce `input` into `self` across contracted axes, folding
    /// each contracted cell into whatever `self` already holds there.
    ///
    /// Under `LaneShare::Whole` (a register or memory accumulator, not a lane-shared plane
    /// fragment) that existing value is the fold's literal starting point: nothing seeds it on
    /// `self`'s behalf. The caller must pre-seed `self` with `inst`'s identity ([`Tile::zero`] for
    /// `Sum`, [`Tile::init`] with `E::min_value()`/`E::max_value()` for `Max`/`Min`) before the
    /// first call, or an uninitialized/stale accumulator folds against garbage.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>, #[comptime] inst: ReduceLeafKind) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => reduce_leaf(self, input, inst),
            Partitioner::Level(level) => {
                let op_space = self.reduce_op_space(input);
                match comptime!(level.schedule()) {
                    Schedule::Direct => self.reduce_direct(input, inst, op_space),
                    Schedule::Staged => self.reduce_staged(input, inst, op_space),
                    Schedule::DoubleBuffered => self.reduce_double(input, inst, op_space),
                }
            }
        }
    }

    /// The level's operation space: the input operand's space, sized by whichever operand
    /// witnesses each dynamic axis.
    fn reduce_op_space<In: Numeric>(&self, input: &Tile<In>) -> Space {
        let merged = comptime!({
            let merged = input.space.clone();
            assert!(
                self.space.axes().all(|axis| merged.contains(axis)),
                "Tile::reduce_axis: the output spans an axis the input does not, \
                 so the walk would never step it and every region would write the same slice"
            );
            merged
        });
        witnessed_space(merged, self, input, input)
    }
}

/// Dispatches the leaf reduction arithmetic at `Partitioner::Final`.
#[cube]
pub fn reduce_leaf<Acc: Numeric, In: Numeric>(
    acc: &mut Tile<Acc>,
    input: &Tile<In>,
    #[comptime] inst: ReduceLeafKind,
) {
    let space = comptime!(acc.space.clone());
    match &mut acc.tile_kind {
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            reduce_register_memory(g, input, space, inst);
        }
        TileKind::PlaneTile(t) => {
            reduce_plane_tile(t, input, space, inst);
        }
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "reduce_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            reduce_plane_tile(&mut t, input, space, inst);
        }
        TileKind::TmaGmem(_) => panic!("reduce: a tma source is not an accumulator sink"),
    }
}

#[cube]
fn reduce_plane_tile<Acc: Numeric, In: Numeric>(
    tile: &mut PlaneTile<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] inst: ReduceLeafKind,
) {
    match tile {
        PlaneTile::Register(d) => {
            reduce_register_data(d, input, acc_space, inst);
        }
        PlaneTile::Cmma(_) | PlaneTile::Mma(_) => {
            panic!(
                "reduce: a hardware mma fragment scatters its rows across lanes in a \
                 layout the elementwise walk cannot address; reduce into a register, \
                 Gmem or Smem accumulator instead"
            );
        }
    }
}

#[cube]
fn reduce_register_data<Acc: Numeric, In: Numeric>(
    acc: &mut RegisterData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] inst: ReduceLeafKind,
) {
    comptime!(assert!(
        inst == ReduceLeafKind::Sum || acc.lane_share == LaneShare::Whole,
        "reduce: a promoted register accumulator only folds partials across lanes with Sum \
         (store_cast_window hardcodes the plane/group fold to sum); a Max or Min reduce under \
         LaneShare::Plane or LaneShare::Group would silently sum the per-lane partials instead"
    ));

    let vw = input.vector_size();
    let size!(V) = vw;
    let pack = input.quant_pack();
    let size!(WP) = comptime!(vw / if pack > 0 { pack } else { 1 });

    if comptime!(pack == 1) {
        reduce_register_data_typed::<Acc, In, i8, WP, V>(acc, input, acc_space, inst);
    } else if comptime!(pack > 1) {
        reduce_register_data_typed::<Acc, In, u32, WP, V>(acc, input, acc_space, inst);
    } else {
        reduce_register_data_typed::<Acc, In, In, WP, V>(acc, input, acc_space, inst);
    }
}

#[cube]
fn reduce_register_data_typed<Acc: Numeric, In: Numeric, I: Numeric, WP: Size, V: Size>(
    acc: &mut RegisterData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] inst: ReduceLeafKind,
) {
    let in_view = input.nd::<I, WP, V>();
    let in_space = comptime!(input.space.clone());
    let vw = input.vector_size();
    let layout = comptime!(ReduceLayout::new(&in_space, &acc_space));
    let total_acc = comptime!(layout.total_acc);

    let count = comptime!(acc.mr * acc.nr);
    comptime!(assert!(
        total_acc == count * acc.vector_size,
        "reduce: RegisterData shape mismatch with accumulator space"
    ));

    #[unroll]
    for a in 0..total_acc {
        let acc_coords = unravel(
            &const_coords(comptime!(layout.acc_extents.clone())),
            comptime!(a as u32),
        );

        let line_idx = comptime!(a / acc.vector_size);
        let lane_idx = comptime!(a % acc.vector_size);
        let seed = acc.data[line_idx].extract(comptime!(lane_idx));

        let curr_val: Acc = reduce_element::<Acc, In, V>(
            &in_view,
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.clone()),
            &acc_coords,
            vw,
            seed,
            inst,
        );

        let mut vec_line = acc.data[line_idx];
        vec_line.insert(comptime!(lane_idx), curr_val);
        acc.data[line_idx] = vec_line;
    }
}

#[cube]
fn reduce_register_memory<Acc: Numeric, In: Numeric>(
    acc: &mut MemData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] inst: ReduceLeafKind,
) {
    let vw = input.vector_size();
    let size!(V) = vw;
    let pack = input.quant_pack();
    let size!(WP) = comptime!(vw / if pack > 0 { pack } else { 1 });

    if comptime!(pack == 1) {
        reduce_register_memory_typed::<Acc, In, i8, WP, V>(acc, input, acc_space, inst);
    } else if comptime!(pack > 1) {
        reduce_register_memory_typed::<Acc, In, u32, WP, V>(acc, input, acc_space, inst);
    } else {
        reduce_register_memory_typed::<Acc, In, In, WP, V>(acc, input, acc_space, inst);
    }
}

#[cube]
fn reduce_register_memory_typed<Acc: Numeric, In: Numeric, I: Numeric, WP: Size, V: Size>(
    acc: &mut MemData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] inst: ReduceLeafKind,
) {
    let in_view = input.nd::<I, WP, V>();
    let in_space = comptime!(input.space.clone());
    let vw = input.vector_size();
    let layout = comptime!(ReduceLayout::new(&in_space, &acc_space));
    let total_acc = comptime!(layout.total_acc);

    // Use the accumulator's native store width so flat_accumulate<W> is a
    // no-op retype (W == vector_size) rather than a pointer retype that the
    // WGSL backend cannot lower.  The pattern mirrors reduce_register_data_typed:
    // outer runtime loop over lines, inner comptime-unrolled loop over lanes.
    let size!(W) = comptime!(acc.store.vector_size);
    // Capture the integer before the mutable borrow so we can reference it inside.
    let ws = comptime!(acc.store.vector_size);
    let total_lines = comptime!(total_acc / acc.store.vector_size);
    comptime!(assert!(
        total_acc % acc.store.vector_size == 0,
        "reduce: MemData total_acc must be divisible by store.vector_size"
    ));
    let mut acc_view = acc.flat_accumulate::<W>();

    for line_idx in 0..total_lines {
        // Seed the whole W-wide line once per accumulator line.
        let seed_vec = acc_view.seed_reduce(line_idx, inst);
        let mut result = seed_vec;

        #[unroll]
        for lane_idx in 0..comptime!(ws) {
            let a = line_idx * comptime!(ws) + comptime!(lane_idx);
            let acc_coords = unravel(
                &const_coords(comptime!(layout.acc_extents.clone())),
                a.fcast::<u32>(),
            );

            let seed = seed_vec.extract(comptime!(lane_idx));
            let curr_val: Acc = reduce_element::<Acc, In, V>(
                &in_view,
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(layout.clone()),
                &acc_coords,
                vw,
                seed,
                inst,
            );

            // insert() is in-place and returns (); mirror reduce_register_data_typed.
            result.insert(comptime!(lane_idx), curr_val);
        }

        acc_view.commit_reduce(line_idx, result, inst);
    }
}

/// The per-element inner reduction shared by both accumulator backings: fold `input` across the
/// contracted axes (`layout.kc` steps) into `seed`, for the single accumulator cell at
/// `acc_coords`. Only the seed/commit around this loop differ between a register block (draining
/// through `RegisterData`'s own lanes) and a memory accumulator (through [`AccumulateView`]).
#[cube]
fn reduce_element<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &MaskedView<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: Space,
    #[comptime] acc_space: Space,
    #[comptime] layout: ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] vw: usize,
    seed: Acc,
    #[comptime] inst: ReduceLeafKind,
) -> Acc {
    let mut curr_val = seed;
    let kc = comptime!(layout.kc);

    for p in 0..kc {
        let reduce_coords = unravel(
            &const_coords(comptime!(layout.reduce_extents.clone())),
            p.fcast::<u32>(),
        );

        let in_coords = resolve_nd_coords(
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.reduce_axes.clone()),
            acc_coords,
            &reduce_coords,
            vw,
            true,
        );

        // `in_view.read`'s own masked fallback is always zero (Sum's identity, not Max's or
        // Min's), so an overhang cell is selected against `inst`'s own identity here instead.
        let valid = in_view.is_in_bounds(in_coords.clone());
        let in_vec = select(
            valid,
            in_view.read(in_coords),
            Vector::<In, V>::cast_from(identity::<In>(inst)),
        );
        let in_lane = resolve_reduce_in_lane(
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.reduce_axes.clone()),
            acc_coords,
            &reduce_coords,
            vw,
        );

        let in_val = in_vec.extract_dynamic(in_lane);
        let in_cast = Acc::cast_from(in_val);

        match comptime!(inst) {
            ReduceLeafKind::Sum => {
                curr_val += in_cast;
            }
            ReduceLeafKind::Max => {
                curr_val = max(curr_val, in_cast);
            }
            ReduceLeafKind::Min => {
                curr_val = min(curr_val, in_cast);
            }
        }
    }

    curr_val
}

/// Comptime bookkeeping shared by both accumulator backings: which axes of `in_space` are
/// contracted against `acc_space`, their extents, the total contracted size (`kc`), and the
/// accumulator's own extents/total cell count.
#[derive(Clone)]
struct ReduceLayout {
    reduce_axes: Vec<Axis>,
    reduce_extents: Vec<usize>,
    kc: usize,
    acc_extents: Vec<usize>,
    total_acc: usize,
}

impl ReduceLayout {
    fn new(in_space: &Space, acc_space: &Space) -> Self {
        let reduce_axes = Space::contracted(&[in_space], acc_space).to_vec();
        let reduce_extents = reduce_axes
            .iter()
            .map(|&a| in_space.extent(a))
            .collect::<Vec<usize>>();
        let kc = in_space.contracted_extent(acc_space);
        let acc_extents = (0..acc_space.rank())
            .map(|p| acc_space.extent_at(p))
            .collect::<Vec<usize>>();
        let total_acc = acc_extents.iter().product::<usize>();
        Self {
            reduce_axes,
            reduce_extents,
            kc,
            acc_extents,
            total_acc,
        }
    }
}

/// The lane within the vectorized line for the input's fastest (innermost) axis, whether it is
/// contracted (in `reduce_axes`) or surviving (in `acc_space`).
#[cube]
fn resolve_reduce_in_lane(
    #[comptime] in_space: Space,
    #[comptime] acc_space: Space,
    #[comptime] reduce_axes: Vec<Axis>,
    acc_coords: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] width: usize,
) -> usize {
    if comptime!(width <= 1) {
        0usize.runtime()
    } else {
        let in_rank = comptime!(in_space.rank());
        let fastest_axis = comptime!(in_space.axis_at(in_rank - 1));
        let raw_coord = if comptime!(acc_space.contains(fastest_axis)) {
            let pos = comptime!(acc_space.position(fastest_axis));
            acc_coords.at(comptime!(pos))
        } else {
            let pos = comptime!(reduce_axes.iter().position(|&r| r == fastest_axis).unwrap());
            reduce_coords.at(comptime!(pos))
        };
        raw_coord.frem(comptime!(width as u32)).fcast::<usize>()
    }
}
