//! Register-resident leaf GEMM microkernel over memory tiles.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

/// Maximum cell count for unrolling register blocks to avoid optimizer overflow.
const UNROLL_BLOCK: usize = 64;

/// Runs the register microkernel over batch matrices for plain or quantized operands.
///
/// Dispatches to 2D direct or N-D gather microkernels based on contracted axes count.
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
) {
    let size!(L) = lhs.vector_size();
    let size!(V) = rhs.vector_size();

    // Validate quantization packing (at most one operand quantized).
    let pack_l = lhs.quant_pack();
    let pack_r = rhs.quant_pack();
    comptime!(assert!(
        pack_l == 0 || pack_r == 0,
        "register leaf: both operands quantized is not a supported direct-serve case"
    ));

    let nd = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space).len() > 1);
    if nd {
        if comptime!(pack_l == 1) {
            mma_register_gather::<E, EL, i8, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
        } else if comptime!(pack_l > 1) {
            mma_register_gather::<E, EL, u32, L, ER, ER, V>(acc, lhs, rhs, space, pack_l, 1usize);
        } else if comptime!(pack_r == 1) {
            mma_register_gather::<E, EL, EL, L, ER, i8, V>(acc, lhs, rhs, space, 1usize, 1usize);
        } else if comptime!(pack_r > 1) {
            mma_register_gather::<E, EL, EL, L, ER, u32, V>(acc, lhs, rhs, space, 1usize, pack_r);
        } else {
            mma_register_gather::<E, EL, EL, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
        }
    } else if comptime!(pack_l == 1) {
        mma_register_direct::<E, EL, i8, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
    } else if comptime!(pack_l > 1) {
        mma_register_direct::<E, EL, u32, L, ER, ER, V>(acc, lhs, rhs, space, pack_l, 1usize);
    } else if comptime!(pack_r == 1) {
        mma_register_direct::<E, EL, EL, L, ER, i8, V>(acc, lhs, rhs, space, 1usize, 1usize);
    } else if comptime!(pack_r > 1) {
        mma_register_direct::<E, EL, EL, L, ER, u32, V>(acc, lhs, rhs, space, 1usize, pack_r);
    } else {
        mma_register_direct::<E, EL, EL, L, ER, ER, V>(acc, lhs, rhs, space, 1usize, 1usize);
    }
}

/// 2D register microkernel for single-axis contraction over batch matrices.
#[cube]
fn mma_register_direct<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    V: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] pack_l: usize,
    #[comptime] pack_r: usize,
) {
    // Ensure single contracted axis and un-gathered operands.
    comptime!(assert!(
        Space::contracted(&[&lhs.space, &rhs.space], &space).len() == 1,
        "register leaf: the 2-D microkernel contracts exactly one axis"
    ));
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered,
        "register leaf: a gathered operand has no 2-D matrix view; it needs the N-D nest"
    ));

    // Compute tile dimensions and vector sizes.
    let lw = lhs.vector_size();
    let vw = rhs.vector_size();
    let (mr, nr, kc) = comptime! {
        (
            space.extent_at(space.rank() - 2),
            space.extent_at(space.rank() - 1) / vw,
            rhs.space.extent_at(rhs.space.rank() - 2)
        )
    };
    let size!(WPL) = comptime!(lw / pack_l);
    let size!(WPR) = comptime!(vw / pack_r);

    let matrices = comptime! {
        let mut count = 1;
        for p in 0..space.rank() - 2 {
            count *= space.extent_at(p);
        }
        count
    };

    for mat in 0..matrices {
        let lhs = lhs.matrix_transparent::<IL, WPL, L>(mat);
        let rhs = rhs.matrix_transparent::<IR, WPR, V>(mat);
        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));

        // Unroll when block size is within limits and unmasked.
        let lhs_check = lhs.check();
        let rhs_check = rhs.check();
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs_check && !rhs_check && !acc_check);
        let mut c = load_accumulators(&mut acc, comptime!(mr), comptime!(nr), unroll);

        // Outer-product updates over contracted K dimension.
        for p in 0..kc {
            let mut b = Array::<Vector<E, V>>::new(nr);
            #[unroll(unroll)]
            for n in 0..nr {
                b[n] = Vector::<E, V>::cast_from(rhs.read((p as u32, n as u32)));
            }
            #[unroll(unroll)]
            for i in 0..mr {
                let lhs_line = lhs.read((i as u32, (p / lw) as u32));
                let a = Vector::<E, V>::cast_from(lhs_line.extract(p % lw));
                #[unroll(unroll)]
                for n in 0..nr {
                    // Explicit FMA to force fused multiply-add instructions.
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }

        store_accumulators(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
    }
}

/// N-D register microkernel for multi-axis contraction or gathered operands.
#[cube]
fn mma_register_gather<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    V: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] pack_l: usize,
    #[comptime] pack_r: usize,
) {
    let lw = lhs.vector_size();
    let vw = rhs.vector_size();
    let size!(WPL) = comptime!(lw / pack_l);
    let size!(WPR) = comptime!(vw / pack_r);

    let rank = comptime!(space.rank());
    let (mr, nr) = comptime!((space.extent_at(rank - 2), space.extent_at(rank - 1) / vw));
    let matrices = comptime!((0..rank - 2).map(|p| space.extent_at(p)).product::<usize>());

    // Flatten contracted axes into total reduction steps.
    let merged = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
    let reduce = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &space).to_vec());
    let reduce_extents = comptime!(reduce.iter().map(|&a| merged.extent(a)).collect::<Vec<_>>());
    let kc = comptime!(reduce_extents.iter().product::<usize>());

    comptime!(assert_operand_shapes(
        &lhs.space, &rhs.space, &space, &reduce, lw
    ));

    // N-D view over operand logical bounds.
    let lhs_view = lhs.nd::<IL, WPL, L>();
    let rhs_view = rhs.nd::<IR, WPR, V>();

    for mat in 0..matrices {
        // Unravel flat matrix index into batch coordinates.
        let mut batch = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(rank - 2) {
            let stride = comptime!(
                ((p + 1)..(rank - 2))
                    .map(|q| space.extent_at(q))
                    .product::<usize>()
            );
            batch.push(
                mat.fcast::<u32>()
                    .fdiv(comptime!(stride as u32))
                    .frem(comptime!(space.extent_at(p) as u32)),
            );
        }

        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));

        // Unroll when block size is within limits and unmasked.
        let lhs_check = lhs_view.check();
        let rhs_check = rhs_view.check();
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= UNROLL_BLOCK && !lhs_check && !rhs_check && !acc_check);
        let mut c = load_accumulators(&mut acc, comptime!(mr), comptime!(nr), unroll);

        // Outer-product updates over N-D contracted axes.
        for p in 0..kc {
            let reduce_coords = unravel_reduce_index(p, comptime!(reduce_extents.clone()));
            let lane = reduce_coords
                .at(comptime!(reduce.len() - 1))
                .frem(comptime!(lw as u32))
                .fcast::<usize>();

            let mut b = Array::<Vector<E, V>>::new(nr);
            #[unroll(unroll)]
            for n in 0..nr {
                let pos = resolve_nd_coords(
                    comptime!(rhs.space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    &batch,
                    &reduce_coords,
                    0u32,
                    n as u32,
                    vw,
                );
                b[n] = Vector::<E, V>::cast_from(rhs_view.read(pos));
            }
            #[unroll(unroll)]
            for i in 0..mr {
                let pos = resolve_nd_coords(
                    comptime!(lhs.space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    &batch,
                    &reduce_coords,
                    i as u32,
                    0u32,
                    lw,
                );
                let a = Vector::<E, V>::cast_from(lhs_view.read(pos).extract(lane));
                #[unroll(unroll)]
                for n in 0..nr {
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }

        store_accumulators(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
    }
}

/// Loads accumulator block into registers.
#[cube]
fn load_accumulators<E: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            c[i * nr + n] = acc.seed((i as u32, n as u32));
        }
    }
    c
}

/// Stores accumulator block to memory.
#[cube]
fn store_accumulators<E: Numeric, V: Size>(
    acc: &mut AccumulateView<'_, E, V>,
    c: Array<Vector<E, V>>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
) {
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            acc.commit((i as u32, n as u32), c[i * nr + n]);
        }
    }
}

/// Unravels a flat reduction index into multi-axis coordinates.
#[cube]
fn unravel_reduce_index(k_step: usize, #[comptime] reduce_extents: Vec<usize>) -> Coords<u32> {
    let n = comptime!(reduce_extents.len());
    let mut reduce_coords = Coords::<u32>::new();

    #[unroll]
    for j in 0..n {
        let stride = comptime!(reduce_extents[(j + 1)..].iter().product::<usize>());
        let axis_coord = k_step.fcast::<u32>().fdiv(comptime!(stride as u32));
        reduce_coords.push(if comptime!(j == 0) {
            axis_coord
        } else {
            axis_coord.frem(comptime!(reduce_extents[j] as u32))
        });
    }

    reduce_coords
}

/// Resolves N-D coordinates for operand access at the current step.
#[cube]
fn resolve_nd_coords(
    #[comptime] operand: Space,
    #[comptime] acc: Space,
    #[comptime] reduce: Vec<Axis>,
    batch: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] width: usize,
) -> CoordsDyn {
    let operand_rank = comptime!(operand.rank());
    let acc_rank = comptime!(acc.rank());
    let mut out = CoordsDyn::new();

    #[unroll]
    for axis_idx in 0..operand_rank {
        let axis = comptime!(operand.axis_at(axis_idx));
        let axis_coord = if comptime!(axis == acc.axis_at(acc_rank - 2)) {
            row
        } else if comptime!(axis == acc.axis_at(acc_rank - 1)) {
            col
        } else if comptime!(reduce.contains(&axis)) {
            let reduce_coord =
                reduce_coords.at(comptime!(reduce.iter().position(|&r| r == axis).unwrap()));
            if comptime!(axis_idx == operand_rank - 1 && width > 1) {
                reduce_coord.fdiv(comptime!(width as u32))
            } else {
                reduce_coord
            }
        } else {
            batch.at(comptime!(acc.position(axis)))
        };
        out.push(axis_coord);
    }

    out
}

/// Validates layout constraints for gather microkernel.
fn assert_operand_shapes(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    reduce: &[Axis],
    lhs_vec_len: usize,
) {
    assert!(
        !reduce.is_empty(),
        "gather leaf: the operands contract no axis against the accumulator"
    );
    let fastest = reduce[reduce.len() - 1];
    assert!(
        lhs.axis_at(lhs.rank() - 1) == fastest,
        "gather leaf: the lhs must line along the fastest contracted axis {fastest:?}"
    );
    assert!(
        lhs_vec_len == 1 || lhs.extent(fastest).is_multiple_of(lhs_vec_len),
        "gather leaf: the lhs's line width {lhs_vec_len} must divide its fastest contracted axis's extent"
    );
    assert!(
        rhs.axis_at(rhs.rank() - 1) == acc.axis_at(acc.rank() - 1),
        "gather leaf: the rhs must line along the accumulator's innermost axis"
    );
}
