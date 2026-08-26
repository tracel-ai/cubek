//! The register nest behind [`Tile::reduce_axis`](crate::Tile::reduce_axis): fold an input's
//! contracted axes into one accumulator cell at a time, under a [`Monoid`].
//!
//! The contraction nest's sibling ([`contract`](super::contract)): same seed, walk `kc`, commit
//! shape over an [`AccumulateView`], one operand and an elementwise fold
//! instead of two operands and a rank-1 outer product. The dispatch that reaches here by
//! accumulator storage stays with the verb, in `ops/reduce/lower.rs`.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::instruction::registers::horizontal;
use crate::*;

#[cube]
pub(crate) fn register_data<Acc: Numeric, In: Numeric>(
    acc: &mut RegisterData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    // The block was built to fold one way and is drained that way; folding it another here would
    // combine the lanes under an operator the partials were never built for.
    comptime!(assert!(
        acc.monoid == monoid,
        "reduce: this accumulator folds under {:?} (stated at `Tile::accumulate`) but is being \
         reduced under {monoid:?}",
        acc.monoid
    ));

    let size!(V) = input.vector_size();
    register_data_body::<Acc, In, V>(acc, input, acc_space, monoid);
}

#[cube]
fn register_data_body<Acc: Numeric, In: Numeric, V: Size>(
    acc: &mut RegisterData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    let in_space = comptime!(input.space.clone());
    let vw = input.vector_size();
    let layout = comptime!(ReduceLayout::new(&in_space, &acc_space));
    let total_acc = comptime!(layout.total_acc);

    let count = comptime!(acc.mr * acc.nr);
    comptime!(assert!(
        total_acc == count * acc.vector_size,
        "reduce: RegisterData shape mismatch with accumulator space"
    ));
    let in_view = input.nd_packed::<V>(comptime!(Guard::Checked));

    #[unroll]
    for a in 0..total_acc {
        let acc_coords = unravel(
            &const_coords(comptime!(layout.acc_extents.clone())),
            comptime!(a as u32),
        );

        let line_idx = comptime!(a / acc.vector_size);
        let lane_idx = comptime!(a % acc.vector_size);
        let seed = acc.data[line_idx].extract(comptime!(lane_idx));

        let curr_val: Acc = element::<Acc, In, V>(
            &in_view,
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.clone()),
            &acc_coords,
            vw,
            seed,
            monoid,
        );

        let mut vec_line = acc.data[line_idx];
        vec_line.insert(comptime!(lane_idx), curr_val);
        acc.data[line_idx] = vec_line;
    }
}

#[cube]
pub(crate) fn memory<Acc: Numeric, In: Numeric>(
    acc: &mut MemData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    let size!(V) = input.vector_size();
    memory_body::<Acc, In, V>(acc, input, acc_space, monoid);
}

#[cube]
fn memory_body<Acc: Numeric, In: Numeric, V: Size>(
    acc: &mut MemData<Acc>,
    input: &Tile<In>,
    #[comptime] acc_space: Space,
    #[comptime] monoid: Monoid,
) {
    let in_space = comptime!(input.space.clone());
    let vw = input.vector_size();
    let layout = comptime!(ReduceLayout::new(&in_space, &acc_space));
    let total_acc = comptime!(layout.total_acc);

    let ws = comptime!(acc.store.vector_size);
    // Preserve the native store width: retyping it as another vector width cannot lower on WGSL.
    let size!(W) = ws;
    comptime!(assert!(
        total_acc % ws == 0,
        "reduce: MemData total_acc must be divisible by store.vector_size"
    ));
    let total_lines = comptime!(total_acc / ws);
    let mut acc_view = acc.flat_accumulate::<W>(monoid);
    let in_view = input.nd_packed::<V>(comptime!(Guard::Checked));

    for line_idx in 0..total_lines {
        let seed_vec = acc_view.seed(line_idx);
        let mut result = seed_vec;

        #[unroll]
        for lane_idx in 0..comptime!(ws) {
            let a = line_idx * comptime!(ws) + comptime!(lane_idx);
            let acc_coords = unravel(
                &const_coords(comptime!(layout.acc_extents.clone())),
                a.fcast::<u32>(),
            );

            let seed = seed_vec.extract(comptime!(lane_idx));
            let curr_val: Acc = element::<Acc, In, V>(
                &in_view,
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(layout.clone()),
                &acc_coords,
                vw,
                seed,
                monoid,
            );

            result.insert(comptime!(lane_idx), curr_val);
        }

        acc_view.commit(line_idx, result);
    }
}

/// The per-element inner reduction shared by both accumulator backings: fold `in_view` across the
/// contracted axes into `seed`, for the single accumulator cell at `acc_coords`.
///
/// A step consumes [`Space::served`] values: past one the input's line runs along the fastest
/// contracted axis, so the whole line folds into this one cell ([`element_lines`]) instead of one
/// scalar at a time ([`element_scalars`]).
#[cube]
fn element<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &MaskedView<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: Space,
    #[comptime] acc_space: Space,
    #[comptime] layout: ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] vw: usize,
    seed: Acc,
    #[comptime] monoid: Monoid,
) -> Acc {
    let served = comptime!(in_space.served(&layout.reduce_axes, vw));
    if comptime!(served > 1) {
        element_lines::<Acc, In, V>(
            in_view,
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.clone()),
            acc_coords,
            served,
            seed,
            monoid,
        )
    } else {
        element_scalars::<Acc, In, V>(
            in_view,
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.clone()),
            acc_coords,
            vw,
            seed,
            monoid,
        )
    }
}

/// [`element`]'s line path: the flat reduce index steps by `served`, so each step lands on a line
/// start and one read serves `served` folds. The lanes accumulate in parallel and collapse
/// through [`horizontal::vector`] once, after the walk.
#[cube]
#[allow(clippy::too_many_arguments)]
fn element_lines<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &MaskedView<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: Space,
    #[comptime] acc_space: Space,
    #[comptime] layout: ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] served: usize,
    seed: Acc,
    #[comptime] monoid: Monoid,
) -> Acc {
    let mut acc_vec = Vector::<Acc, V>::cast_from(Monoid::identity::<Acc>(monoid));
    for p in 0..comptime!(layout.kc / served) {
        let reduce_coords = unravel(
            &const_coords(comptime!(layout.reduce_extents.clone())),
            (p * comptime!(served)).fcast::<u32>(),
        );

        let in_coords = resolve_nd_coords(
            comptime!(in_space.clone()),
            comptime!(acc_space.clone()),
            comptime!(layout.reduce_axes.clone()),
            acc_coords,
            &reduce_coords,
            served,
            true,
        );

        // Same masking as the scalar path, a whole line at a time.
        let in_vec = match comptime!(monoid) {
            Monoid::Sum => in_view.read(in_coords),
            Monoid::Prod | Monoid::Max | Monoid::Min => {
                let valid = in_view.is_in_bounds(in_coords.clone());
                select(
                    valid,
                    in_view.read(in_coords),
                    Vector::<In, V>::cast_from(Monoid::identity::<In>(monoid)),
                )
            }
        };
        acc_vec = monoid.fold::<Vector<Acc, V>>(acc_vec, Vector::<Acc, V>::cast_from(in_vec));
    }
    monoid.fold::<Acc>(seed, horizontal::vector::<Acc, V>(acc_vec, served, monoid))
}

/// [`element`]'s scalar path: one extract per contracted element.
#[cube]
#[allow(clippy::too_many_arguments)]
fn element_scalars<Acc: Numeric, In: Numeric, V: Size>(
    in_view: &MaskedView<'_, Vector<In, V>, CoordsDyn>,
    #[comptime] in_space: Space,
    #[comptime] acc_space: Space,
    #[comptime] layout: ReduceLayout,
    acc_coords: &Coords<u32>,
    #[comptime] vw: usize,
    seed: Acc,
    #[comptime] monoid: Monoid,
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

        // Memory reads already return Sum's zero identity out of bounds; procedural reads are
        // always valid. The others need the identity read in explicitly.
        let in_vec = match comptime!(monoid) {
            Monoid::Sum => in_view.read(in_coords),
            Monoid::Prod | Monoid::Max | Monoid::Min => {
                let valid = in_view.is_in_bounds(in_coords.clone());
                select(
                    valid,
                    in_view.read(in_coords),
                    Vector::<In, V>::cast_from(Monoid::identity::<In>(monoid)),
                )
            }
        };
        let in_val = if comptime!(vw <= 1) {
            in_vec.extract(0usize)
        } else {
            let in_lane = resolve_reduce_in_lane(
                comptime!(in_space.clone()),
                comptime!(acc_space.clone()),
                comptime!(layout.reduce_axes.clone()),
                acc_coords,
                &reduce_coords,
                vw,
            );
            in_vec.extract_dynamic(in_lane)
        };
        let in_cast = Acc::cast_from(in_val);

        curr_val = monoid.fold::<Acc>(curr_val, in_cast);
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
        let kc = reduce_extents.iter().product::<usize>();
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
