//! K-major general N-D gather, with operand reads hoisted to their widest valid reuse scope.

use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::microkernel::block::{self, lane_component};
use crate::*;

use super::coords::cell_read;

#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<
    E: Numeric,
    EL: Numeric,
    IL: Numeric,
    L: Size,
    ER: Numeric,
    IR: Numeric,
    WPL: Size,
    WPR: Size,
    V: Size,
>(
    acc: &mut MemData<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] space: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] reduce_extents: Vec<usize>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] matrices: usize,
    #[comptime] kc: usize,
    #[comptime] lw: usize,
    #[comptime] vw: usize,
    #[comptime] lhs_spans_col: bool,
    #[comptime] rhs_spans_row: bool,
    #[comptime] rhs_spans_col: bool,
    #[comptime] config: MemoryMmaConfig,
) {
    let rank = comptime!(space.rank());
    let lhs_view = lhs.nd::<IL, WPL, L>();
    let rhs_view = rhs.nd::<IR, WPR, V>();
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);
    let k_lines = comptime!(kc / lw);
    let k_tail = comptime!(kc % lw);
    let unroll_limit = comptime!(config.unroll_limit);
    let lane_fanout = comptime!(config.lane_fanout);
    let batch_extents = const_coords(comptime!(
        (0..rank - 2)
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    ));

    for mat in 0..matrices {
        let batch = unravel(&batch_extents, mat.fcast::<u32>());
        let mut acc = acc.matrix_accumulate::<V>(mat, comptime!(space.clone()));
        let acc_check = acc.check();
        let unroll = comptime!(mr * nr <= unroll_limit && !lhs_check && !rhs_check && !acc_check);
        let mut c = block::seed(&mut acc, comptime!(mr), comptime!(nr), unroll);
        let mut b = Array::<Vector<E, V>>::new(nr);

        if comptime!(lane_fanout && lw > 1) {
            for line in 0..k_lines {
                #[unroll]
                for lane in 0..lw {
                    rank1_update::<E, EL, L, ER, V>(
                        &lhs_view,
                        &rhs_view,
                        &mut c,
                        &mut b,
                        &batch,
                        line * lw + lane,
                        comptime!(Some(lane)),
                        mr,
                        nr,
                        unroll,
                        comptime!(space.clone()),
                        comptime!(reduce.clone()),
                        comptime!(reduce_extents.clone()),
                        comptime!(lhs.space.clone()),
                        comptime!(rhs.space.clone()),
                        lw,
                        vw,
                        lhs_spans_col,
                        rhs_spans_row,
                        rhs_spans_col,
                    );
                }
            }
            #[unroll]
            for lane in 0..k_tail {
                rank1_update::<E, EL, L, ER, V>(
                    &lhs_view,
                    &rhs_view,
                    &mut c,
                    &mut b,
                    &batch,
                    comptime!(k_lines * lw + lane),
                    comptime!(Some(lane)),
                    mr,
                    nr,
                    unroll,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lw,
                    vw,
                    lhs_spans_col,
                    rhs_spans_row,
                    rhs_spans_col,
                );
            }
        } else {
            for p in 0..kc {
                rank1_update::<E, EL, L, ER, V>(
                    &lhs_view,
                    &rhs_view,
                    &mut c,
                    &mut b,
                    &batch,
                    p,
                    comptime!(None),
                    mr,
                    nr,
                    unroll,
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    comptime!(reduce_extents.clone()),
                    comptime!(lhs.space.clone()),
                    comptime!(rhs.space.clone()),
                    lw,
                    vw,
                    lhs_spans_col,
                    rhs_spans_row,
                    rhs_spans_col,
                );
            }
        }

        block::commit(&mut acc, c, comptime!(mr), comptime!(nr), unroll);
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs_view: &MaskedView<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &MaskedView<'_, Vector<ER, V>, CoordsDyn>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    batch: &Coords<u32>,
    p: usize,
    #[comptime] lane: Option<usize>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
    #[comptime] space: Space,
    #[comptime] reduce: Vec<Axis>,
    #[comptime] reduce_extents: Vec<usize>,
    #[comptime] lhs_space: Space,
    #[comptime] rhs_space: Space,
    #[comptime] lw: usize,
    #[comptime] vw: usize,
    #[comptime] lhs_spans_col: bool,
    #[comptime] rhs_spans_row: bool,
    #[comptime] rhs_spans_col: bool,
) {
    let reduce_coords = unravel(&const_coords(reduce_extents), p.fcast::<u32>());

    if comptime!(!rhs_spans_row) {
        #[unroll(unroll)]
        for n in 0..nr {
            b[n] = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                0u32,
                n as u32,
                &reduce_coords,
                comptime!(rhs_space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                vw,
            ));
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        let mut a_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(!lhs_spans_col) {
            let line = cell_read::<EL, L>(
                lhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(lhs_space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                lw,
            );
            a_row = lane_component::<E, EL, L, V>(line, p, lane, lw);
        }
        let mut b_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(rhs_spans_row && !rhs_spans_col) {
            b_row = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(rhs_space.clone()),
                comptime!(space.clone()),
                comptime!(reduce.clone()),
                vw,
            ));
        }
        #[unroll(unroll)]
        for n in 0..nr {
            let a = if comptime!(lhs_spans_col) {
                let line = cell_read::<EL, L>(
                    lhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(lhs_space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    lw,
                );
                lane_component::<E, EL, L, V>(line, p, lane, lw)
            } else {
                a_row
            };
            let v = if comptime!(!rhs_spans_row) {
                b[n]
            } else if comptime!(!rhs_spans_col) {
                b_row
            } else {
                Vector::<E, V>::cast_from(cell_read::<ER, V>(
                    rhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(rhs_space.clone()),
                    comptime!(space.clone()),
                    comptime!(reduce.clone()),
                    vw,
                ))
            };
            c[i * nr + n] = fma(a, v, c[i * nr + n]);
        }
    }
}
