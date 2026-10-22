use super::base::PlaneUnitGroupAttentionMicrokernel;
use crate::attention::unit::unit_attention_step;
use crate::attention::{
    B, CubePlaneAttention, CubePlaneAttentionExpand, CubePlaneLayout, DV, G, KV, Q, S,
    StagedPlaneOperands,
};
use core::marker::PhantomData;

use cubecl::tile::{Region, Semiring, StageStorage, Tile, Walk, ops::softmax::OnlineSoftmax};
use cubecl::{self as cubecl, cube, prelude::*};

/// A cube's plane unit groups over the cube's stages: each unit's sum of its rows' outputs in a
/// register block held across the walk, and the softmax state of its rows.
#[derive(CubeType)]
pub struct PlaneUnitGroupAttention<EA: Float, ES: Float> {
    sum: Tile,
    softmax: OnlineSoftmax<EA>,
    query: Tile,
    /// The element the keys and values are staged at, which a unit's step reads them in.
    #[cube(comptime)]
    staged: PhantomData<ES>,
    #[cube(comptime)]
    microkernel: PlaneUnitGroupAttentionMicrokernel,
}

#[cube]
impl<EA: Float, E: Float, ES: Float> CubePlaneAttention<EA, E, Tile, Tile, Tile>
    for PlaneUnitGroupAttention<EA, ES>
{
    type Microkernel = PlaneUnitGroupAttentionMicrokernel;
    type Opening = StagedPlaneOperands;
    type Line = Vector<EA, Const<1>>;

    /// Its sum opened at zero on the cube's output, in the value's register blocks.
    fn open(
        _walk: &Walk,
        opening: &StagedPlaneOperands,
        #[comptime] microkernel: PlaneUnitGroupAttentionMicrokernel,
        #[comptime] _layout: CubePlaneLayout,
    ) -> Self {
        let mut sum = opening.out.accumulator::<EA>(
            &opening.bias,
            &opening.value,
            comptime!(microkernel.unit.value.tile_instruction()),
            Semiring::SUM_PROD,
        );
        sum.zero();
        PlaneUnitGroupAttention::<EA, ES> {
            sum,
            softmax: OnlineSoftmax::<EA>::new(comptime!(microkernel.rows()), S),
            query: opening.query.clone(),
            staged: PhantomData,
            microkernel,
        }
    }

    /// [`unit_attention_step()`] on each unit's rows of the plane's region of the stage.
    fn step(&mut self, plane: &Region, keys: &Tile, values: &Tile, visible: &Tile, scale: EA) {
        for rows in plane {
            unit_attention_step::<EA>(
                &rows,
                &mut self.sum.at(&rows),
                &mut self.softmax,
                &self.query.at(&rows),
                &keys.at(&rows),
                &values.at(&rows),
                &visible.at(&rows),
                scale,
                comptime!(self.microkernel.unit),
            );
        }
    }

    /// Each unit's rows divided by their sums and drained into `out`. Its candidates hold no
    /// planes along the keys, so no rows are merged.
    fn store_normalized(&mut self, cube: &Region, out: &Tile) {
        for plane in cube.planes() {
            for rows in plane {
                let sum = self.sum.at(&rows);
                sum.along(DV).mul(&self.softmax.recip_l());
                sum.drained_into(&out.at(&rows));
            }
        }
    }

    /// Each unit's sum drained, as it stands, into its window of a scratch of the walk's step,
    /// then its rows copied from there into their slots by the unit alone, beside its rows'
    /// softmax state.
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        walk: &Walk,
        plane: &Region,
        slot_out: &mut [Vector<EA, Const<1>>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    ) {
        // Over the output's axes, as the register blocks it drains are laid.
        let window = Tile::scratch::<EA>(
            walk,
            comptime!(vec![B, KV, G, Q, DV]),
            StageStorage::Strided,
        );
        let state = self.softmax.state();
        for unit_rows in plane {
            let rows_window = window.at(&unit_rows);
            self.sum.at(&unit_rows).drained_into(&rows_window);
            let value = comptime!(rows_window.space().extent(DV));
            let read = rows_window.flat::<EA, Const<1>>();
            let first_row = unit_rows.origin(Q) - plane.origin(Q);
            #[unroll]
            for r in 0..comptime!(self.microkernel.rows()) {
                let row = first_row + r;
                if row < rows {
                    let slot = first_slot + row * row_stride;
                    for d in 0..value {
                        slot_out[slot * value + d] = read.read(r * value + d);
                    }
                    slot_max[slot] = state.max[r];
                    slot_sum[slot] = state.sum[r];
                }
            }
        }
    }
}
