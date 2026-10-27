use super::base::PlaneShuffleAttentionMicrokernel;
use crate::attention::{
    CubePlaneAttention, CubePlaneAttentionExpand, CubePlaneLayout, CubePlaneMicrokernel, D,
    GlobalPlaneOperands, Q, S, StepLines, StepLinesExpand, StepWalk, VisibleKeys,
};
use cubecl::tile::{Monoid, Region, Tile, Walk, space::UnitShare};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's rows in its units' registers while it walks the keys: each unit's lines of every
/// row's query and unnormalized output, and its team's running max and sum of every row.
///
/// A unit is team `unit / units_per_key` and holds lines `unit % units_per_key + i · units_per_key`
/// of each row, of the query and of the output alike.
#[derive(CubeType)]
pub struct PlaneShuffleAttention<EA: Float, N: Size> {
    query: Array<Vector<EA, N>>,
    sum: Array<Vector<EA, N>>,
    max: Array<EA>,
    total: Array<EA>,
    #[cube(comptime)]
    microkernel: PlaneShuffleAttentionMicrokernel,
    /// Elements of a line, the operands' one width.
    #[cube(comptime)]
    width: usize,
    /// Lines of a row of the query.
    #[cube(comptime)]
    head_lines: usize,
    /// Lines of a row of the output.
    #[cube(comptime)]
    value_lines: usize,
}

#[cube]
impl<EA: Float, N: Size> PlaneShuffleAttention<EA, N> {
    /// The plane's rows of `query`, windowed to them, each unit loading its lines of every row.
    fn load_query<E: Float>(&mut self, query: &Tile) {
        let microkernel = comptime!(self.microkernel);
        let head_lines = comptime!(self.head_lines);
        let units_per_key = comptime!(microkernel.settings.units_per_key.get());
        let per_head = comptime!(head_lines.div_ceil(units_per_key));
        let lane = UNIT_POS_PLANE as usize % units_per_key;
        let read = query.flat::<E, N>();
        #[unroll]
        for r in 0..comptime!(microkernel.seq_q()) {
            #[unroll]
            for p in 0..per_head {
                let line = lane + p * units_per_key;
                let mut held = Vector::<EA, N>::cast_from(0u32);
                if line < head_lines {
                    held = Vector::cast_from(read.read(r * head_lines + line));
                }
                self.query[r * per_head + p] = held;
            }
        }
    }

    /// The plane's teams merged into one state of its rows: every unit leaves holding the plane's
    /// max and sum of each row, and its lines of the output summed over the teams, each team's
    /// rescaled to the merged max. Run once, after the last step.
    fn close(&mut self) {
        let microkernel = comptime!(self.microkernel);
        if comptime!(microkernel.teams > 1) {
            let rows = comptime!(microkernel.seq_q());
            let units_per_key = comptime!(microkernel.settings.units_per_key.get());
            let per_value = comptime!(self.value_lines.div_ceil(units_per_key));
            let teams = comptime!(UnitShare::Group {
                unit_bits: (microkernel.units() - 1) & !(units_per_key - 1)
            });
            let size!(W1) = 1usize;
            #[unroll]
            for r in 0..rows {
                let own = self.max[r];
                let merged = teams
                    .reduce::<Vector<EA, W1>>(Vector::cast_from(own), Monoid::Max)
                    .extract(0usize);
                let weight = (own - merged).exp();
                self.total[r] = teams
                    .reduce::<Vector<EA, W1>>(
                        Vector::cast_from(self.total[r] * weight),
                        Monoid::Sum,
                    )
                    .extract(0usize);
                self.max[r] = merged;
                #[unroll]
                for p in 0..per_value {
                    self.sum[r * per_value + p] = teams.reduce::<Vector<EA, N>>(
                        self.sum[r * per_value + p] * Vector::cast_from(weight),
                        Monoid::Sum,
                    );
                }
            }
        }
    }

    /// The closed rows written as a partial, the first `rows` of them: row `r`'s running max and
    /// sum at `first_slot + r · row_stride` of `max` and `sum`, and its unnormalized output's line
    /// `l` at that slot times the lines of a value, plus `l`, of `out` ([`AttentionSlotShape`](crate::attention::AttentionSlotShape)).
    /// The first team's units write their lines, and the plane's first unit the state.
    fn write_partial(
        &self,
        out: &mut [Vector<EA, N>],
        max: &mut [EA],
        sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    ) {
        let microkernel = comptime!(self.microkernel);
        let value_lines = comptime!(self.value_lines);
        let units_per_key = comptime!(microkernel.settings.units_per_key.get());
        let per_value = comptime!(value_lines.div_ceil(units_per_key));
        let unit = UNIT_POS_PLANE as usize;
        #[unroll]
        for r in 0..comptime!(microkernel.seq_q()) {
            if r < rows {
                let slot = first_slot + r * row_stride;
                if unit < units_per_key {
                    #[unroll]
                    for p in 0..per_value {
                        let line = unit + p * units_per_key;
                        if line < value_lines {
                            out[slot * value_lines + line] = self.sum[r * per_value + p];
                        }
                    }
                }
                if unit == 0 {
                    max[slot] = self.max[r];
                    sum[slot] = self.total[r];
                }
            }
        }
    }

    /// The closed rows normalized and written into `out`, windowed to them: the first team's
    /// units write their lines of each row, divided by the row's sum, and a row that saw no key
    /// writes zeros.
    fn store_rows<E: Float>(&self, out: &mut Tile) {
        let microkernel = comptime!(self.microkernel);
        let value_lines = comptime!(self.value_lines);
        let units_per_key = comptime!(microkernel.settings.units_per_key.get());
        let per_value = comptime!(value_lines.div_ceil(units_per_key));
        let unit = UNIT_POS_PLANE as usize;
        let mut lines = out.flat_mut::<E, N>();
        if unit < units_per_key {
            #[unroll]
            for r in 0..comptime!(microkernel.seq_q()) {
                let total = self.total[r];
                let recip = select(
                    total > EA::from_int(0),
                    EA::from_int(1) / total,
                    EA::from_int(0),
                );
                #[unroll]
                for p in 0..per_value {
                    let line = unit + p * units_per_key;
                    if line < value_lines {
                        lines.write(
                            r * value_lines + line,
                            Vector::cast_from(
                                self.sum[r * per_value + p] * Vector::cast_from(recip),
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[cube]
impl<EA: Float, E: Float, N: Size, K: StepLines<N>, L: StepLines<N>>
    CubePlaneAttention<EA, E, K, L, VisibleKeys> for PlaneShuffleAttention<EA, N>
{
    type Microkernel = PlaneShuffleAttentionMicrokernel;
    type Opening = GlobalPlaneOperands;
    type Line = Vector<EA, N>;

    /// Every team's state before any key, the output at zero, and each plane's rows of the query
    /// in its units' registers, read off the walk's first step.
    fn open(
        walk: &Walk,
        opening: &GlobalPlaneOperands,
        #[comptime] microkernel: PlaneShuffleAttentionMicrokernel,
        #[comptime] _layout: CubePlaneLayout,
    ) -> Self {
        let width = opening.query.vector_size();
        let head_lines = comptime!(opening.query.space().extent(D) / width);
        let value_lines = comptime!(opening.val_dim / width);
        let rows = comptime!(microkernel.seq_q());
        let units_per_key = comptime!(microkernel.settings.units_per_key.get());
        let per_head = comptime!(head_lines.div_ceil(units_per_key));
        let per_value = comptime!(value_lines.div_ceil(units_per_key));
        let mut sum = Array::<Vector<EA, N>>::new(comptime!(rows * per_value));
        let mut max = Array::<EA>::new(rows);
        let mut total = Array::<EA>::new(rows);
        #[unroll]
        for r in 0..rows {
            #[unroll]
            for p in 0..per_value {
                sum[r * per_value + p] = Vector::cast_from(0u32);
            }
            max[r] = EA::min_value();
            total[r] = EA::from_int(0);
        }
        let mut shuffle = PlaneShuffleAttention::<EA, N> {
            query: Array::<Vector<EA, N>>::new(comptime!(rows * per_head)),
            sum,
            max,
            total,
            microkernel,
            width,
            head_lines,
            value_lines,
        };
        for plane in walk.planes() {
            shuffle.load_query::<E>(&opening.query.at(&plane));
        }
        shuffle
    }

    /// Each team loads its keys of the step from `keys` and `values` ([`StepLines`]), every load
    /// issued before any is used, a key past the attended ones reading nothing; the team sums
    /// each key's partial dots against each row, takes the step's scores into its rows' max and
    /// sum once, and adds the values they weigh into its lines of the output, rescaled to the new
    /// max. A key a row does not see weighs nothing.
    fn step(&mut self, plane: &Region, keys: &K, values: &L, visible: &VisibleKeys, scale: EA) {
        let microkernel = comptime!(self.microkernel);
        let (head_lines, value_lines) = comptime!((self.head_lines, self.value_lines));
        let rows = comptime!(microkernel.seq_q());
        let teams = comptime!(microkernel.teams);
        let in_flight = comptime!(microkernel.settings.keys_in_flight.get());
        let units_per_key = comptime!(microkernel.settings.units_per_key.get());
        let (per_head, per_value) = comptime!((
            head_lines.div_ceil(units_per_key),
            value_lines.div_ceil(units_per_key)
        ));
        let unit = UNIT_POS_PLANE as usize;
        let (team, lane) = (unit / units_per_key, unit % units_per_key);
        let (first_query, first_key) = (plane.origin(Q), plane.origin(S));
        let walk = StepWalk {
            team,
            lane,
            attended: visible.attended,
            teams,
            keys_in_flight: in_flight,
            units_per_key,
        };
        let key_reads = keys.load(plane, &walk, head_lines);
        let value_reads = values.load(plane, &walk, value_lines);

        let team_share = comptime!(UnitShare::Group {
            unit_bits: units_per_key - 1
        });
        let width = comptime!(self.width);
        let size!(W1) = 1usize;
        let mut weights = Array::<EA>::new(comptime!(rows * in_flight));
        let mut corrections = Array::<EA>::new(rows);
        #[unroll]
        for r in 0..rows {
            let query = (first_query + r) as u32;
            let mut scores = Array::<EA>::new(in_flight);
            #[unroll]
            for c in 0..in_flight {
                let mut partial = EA::from_int(0);
                #[unroll]
                for p in 0..per_head {
                    partial += Monoid::reduce::<EA, N>(
                        self.query[r * per_head + p]
                            * Vector::cast_from(key_reads[c * per_head + p]),
                        width,
                        Monoid::Sum,
                    );
                }
                scores[c] = team_share
                    .reduce::<Vector<EA, W1>>(Vector::cast_from(partial), Monoid::Sum)
                    .extract(0usize)
                    * scale;
            }
            let before = self.max[r];
            let mut after = before;
            #[unroll]
            for c in 0..in_flight {
                if visible.sees(query, (first_key + c * teams + team) as u32) {
                    after = max(after, scores[c]);
                }
            }
            let mut added = EA::from_int(0);
            #[unroll]
            for c in 0..in_flight {
                let seen = visible.sees(query, (first_key + c * teams + team) as u32);
                let weight = select(seen, (scores[c] - after).exp(), EA::from_int(0));
                weights[r * in_flight + c] = weight;
                added += weight;
            }
            let correction = (before - after).exp();
            self.max[r] = after;
            self.total[r] = self.total[r] * correction + added;
            corrections[r] = correction;
        }

        #[unroll]
        for r in 0..rows {
            #[unroll]
            for p in 0..per_value {
                let mut line = self.sum[r * per_value + p] * Vector::cast_from(corrections[r]);
                #[unroll]
                for c in 0..in_flight {
                    line += Vector::<EA, N>::cast_from(value_reads[c * per_value + p])
                        * Vector::cast_from(weights[r * in_flight + c]);
                }
                self.sum[r * per_value + p] = line;
            }
        }
    }

    /// The teams merged (`close`), then each plane's rows divided and written where it holds
    /// them. Planes that split the same rows' keys are never stored this way: the cube merges
    /// them from what each leaves in slots ([`store_partial`](crate::attention::CubePlaneAttention::store_partial)).
    fn store_normalized(&mut self, cube: &Region, out: &Tile) {
        self.close();
        for plane in cube.planes() {
            self.store_rows::<E>(&mut out.at(&plane));
        }
    }

    /// The teams merged (`close`), then the plane's rows written as they stand.
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        _walk: &Walk,
        _plane: &Region,
        slot_out: &mut [Vector<EA, N>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    ) {
        self.close();
        self.write_partial(slot_out, slot_max, slot_sum, first_slot, row_stride, rows);
    }
}
