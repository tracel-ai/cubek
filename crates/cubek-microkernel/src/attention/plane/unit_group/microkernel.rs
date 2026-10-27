use super::base::PlaneUnitGroupAttentionMicrokernel;
use crate::attention::unit::unit_attention_step;
use crate::attention::{B, DV, G, KV, Q, S};
use core::marker::PhantomData;

use cubecl::tile::{Region, Semiring, StageStorage, Tile, Walk, ops::softmax::OnlineSoftmax};
use cubecl::{self as cubecl, cube, prelude::*};

/// Plane unit groups over staged keys and values: each unit's sum of its rows' outputs in a
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
impl<EA: Float, ES: Float> PlaneUnitGroupAttention<EA, ES> {
    /// Every plane's rows before any key: its sum opened at zero on `out`, the output it is shaped
    /// by, in the value's register blocks against `value` and `bias`. `query` is staged.
    pub fn open(
        query: &Tile,
        out: &Tile,
        value: &Tile,
        bias: &Tile,
        #[comptime] microkernel: PlaneUnitGroupAttentionMicrokernel,
    ) -> Self {
        let mut sum = out.accumulator::<EA>(
            bias,
            value,
            comptime!(microkernel.unit.value.tile_instruction()),
            Semiring::SUM_PROD,
        );
        sum.zero();
        PlaneUnitGroupAttention::<EA, ES> {
            sum,
            softmax: OnlineSoftmax::<EA>::new(comptime!(microkernel.unit_seq_q()), S),
            query: query.clone(),
            staged: PhantomData,
            microkernel,
        }
    }

    /// `plane`, its region of one step of the walk, taken into its rows: `unit_attention_step` on
    /// each unit's rows, its keys and values read from `keys` and `values`, each row seeing those
    /// `visible` admits, scaled by `scale`.
    pub fn step(&mut self, plane: &Region, keys: &Tile, values: &Tile, visible: &Tile, scale: EA) {
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

    /// Each unit's rows of `region` divided by their sums and drained into `out`, `region`'s
    /// window of the output. No two planes hold the same rows, so none are merged.
    pub fn store_normalized(&mut self, region: &Region, out: &Tile) {
        for plane in region.planes() {
            for rows in plane {
                let sum = self.sum.at(&rows);
                sum.along(DV).mul(&self.softmax.recip_l());
                sum.drained_into(&out.at(&rows));
            }
        }
    }

    /// Each unit's rows' logsumexp written into `logsumexp`, `region`'s window of it, from their
    /// state, which the unit holds whole.
    pub fn store_logsumexp(&self, region: &Region, logsumexp: &Tile) {
        for plane in region.planes() {
            for rows in plane {
                self.softmax
                    .store_logsumexp(&self.sum.at(&rows), &mut logsumexp.at(&rows));
            }
        }
    }

    /// `plane`'s own rows of `walk`'s first step left undivided in slots, the first `rows` of
    /// them: row `r`'s output, running max and running sum at slot `first_slot + r · row_stride`
    /// of `slot_out`, `slot_max` and `slot_sum`. Each unit's sum is drained, as it stands, into
    /// its window of a scratch of the walk's step, then its rows copied from there into their
    /// slots by the unit alone, beside its rows' softmax state.
    #[allow(clippy::too_many_arguments)]
    pub fn store_partial(
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
            for r in 0..comptime!(self.microkernel.unit_seq_q()) {
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
