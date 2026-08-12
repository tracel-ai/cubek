//! Lowering `c.reduce_axis(input, inst)`: at a final tile, the leaf instruction; while levels remain,
//! walk this level under its [`Schedule`].

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use super::kind::ReduceLeafKind;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.reduce_axis(input, inst)`: reduce `input` into `self` across contracted axes.
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
            let merged = Space::merge(&[&input.space]);
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
    let acc_rank = comptime!(acc_space.rank());
    let vw = input.vector_size();

    let reduce_axes = comptime!(Space::contracted(&[&in_space], &acc_space).to_vec());
    let reduce_extents = comptime!(
        reduce_axes
            .iter()
            .map(|&a| in_space.extent(a))
            .collect::<Vec<usize>>()
    );
    let kc = comptime!(in_space.contracted_extent(&acc_space));

    let acc_extents = comptime!(
        (0..acc_rank)
            .map(|p| acc_space.extent_at(p))
            .collect::<Vec<usize>>()
    );
    let total_acc = comptime!(acc_extents.iter().product::<usize>());

    let count = comptime!(acc.mr * acc.nr);
    comptime!(assert!(
        total_acc == count * acc.vector_size,
        "reduce: RegisterData shape mismatch with accumulator space"
    ));

    #[unroll]
    for a in 0..total_acc {
        let acc_coords = unravel(
            &const_coords(comptime!(acc_extents.clone())),
            comptime!(a as u32),
        );

        let line_idx = comptime!(a / acc.vector_size);
        let lane_idx = comptime!(a % acc.vector_size);
        let mut curr_val = acc.data[line_idx].extract(comptime!(lane_idx));

        for p in 0..kc {
            let reduce_coords = unravel(
                &const_coords(comptime!(reduce_extents.clone())),
                p.fcast::<u32>(),
            );

            let in_coords = resolve_reduce_nd_coords(
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(reduce_axes.clone()),
                &acc_coords,
                &reduce_coords,
                vw,
            );

            let in_vec = in_view.read(in_coords);
            let in_lane = resolve_reduce_in_lane(
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(reduce_axes.clone()),
                &acc_coords,
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
    let acc_rank = comptime!(acc_space.rank());
    let vw = input.vector_size();

    let reduce_axes = comptime!(Space::contracted(&[&in_space], &acc_space).to_vec());
    let reduce_extents = comptime!(
        reduce_axes
            .iter()
            .map(|&a| in_space.extent(a))
            .collect::<Vec<usize>>()
    );
    let kc = comptime!(in_space.contracted_extent(&acc_space));

    let acc_extents = comptime!(
        (0..acc_rank)
            .map(|p| acc_space.extent_at(p))
            .collect::<Vec<usize>>()
    );
    let total_acc = comptime!(acc_extents.iter().product::<usize>());

    let size!(One) = 1usize;
    let mut acc_view = acc.flat_accumulate::<One>();

    for a in 0..total_acc {
        let acc_coords = unravel(
            &const_coords(comptime!(acc_extents.clone())),
            a.fcast::<u32>(),
        );

        let mut curr_val = acc_view.seed_reduce(a, inst).extract(0usize);

        for p in 0..kc {
            let reduce_coords = unravel(
                &const_coords(comptime!(reduce_extents.clone())),
                p.fcast::<u32>(),
            );

            let in_coords = resolve_reduce_nd_coords(
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(reduce_axes.clone()),
                &acc_coords,
                &reduce_coords,
                vw,
            );

            let in_vec = in_view.read(in_coords);
            let in_lane = resolve_reduce_in_lane(
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(reduce_axes.clone()),
                &acc_coords,
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

        acc_view.commit_reduce(a, Vector::<Acc, One>::cast_from(curr_val), inst);
    }
}

/// The coordinate `input` is read at, one entry per axis of its own space. The innermost axis is
/// addressed in lines (matching [`Tile::nd`]), so its coordinate is divided by `width` regardless
/// of whether that axis is reduced or retained in the accumulator.
#[cube]
fn resolve_reduce_nd_coords(
    #[comptime] in_space: Space,
    #[comptime] acc_space: Space,
    #[comptime] reduce_axes: Vec<Axis>,
    acc_coords: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] width: usize,
) -> CoordsDyn {
    let in_rank = comptime!(in_space.rank());
    let mut out = CoordsDyn::new();

    #[unroll]
    for p in 0..in_rank {
        let axis = comptime!(in_space.axis_at(p));
        let raw_coord = if comptime!(acc_space.contains(axis)) {
            let pos = comptime!(acc_space.position(axis));
            acc_coords.at(comptime!(pos))
        } else {
            let pos = comptime!(reduce_axes.iter().position(|&r| r == axis).unwrap());
            reduce_coords.at(comptime!(pos))
        };
        let coord = if comptime!(p == in_rank - 1 && width > 1) {
            raw_coord.fdiv(comptime!(width as u32))
        } else {
            raw_coord
        };
        out.push(coord);
    }

    out
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
