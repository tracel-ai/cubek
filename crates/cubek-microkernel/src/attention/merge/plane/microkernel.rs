use crate::attention::{AttentionSlotBox, AttentionSlotShape};
use cubecl::tile::{Monoid, Tile, space::UnitShare};
use cubecl::{self as cubecl, cube, prelude::*};

/// The slots of `slot_box`'s queries, laid out as `slots` states, merged into those queries of
/// `out` by one plane, a query at a time ([`MergedSlotRow::new`]), the first `queries` of the box.
/// A query whose every slot saw nothing stays zero.
///
/// The slots lie as [`AttentionSlotBox`] states; `out` is the attention's output,
/// `[b, kv, g, q, dv]` at its own strides.
#[cube]
pub fn plane_merge_attention_microkernel<EA: Float, E: Float, N: Size>(
    // Read only, taken as the kernel holds them: cube code passes no `&mut` where a `&` is asked.
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    out: &mut Tensor<E>,
    slot_box: &AttentionSlotBox,
    #[comptime] slots: AttentionSlotShape,
    #[comptime] queries: usize,
) {
    let AttentionSlotShape { units, val_dim, .. } = slots;
    let width = slot_out.vector_size();
    let lines = comptime!(val_dim / width);
    let elements = comptime!(lines.div_ceil(units));
    let lane = UNIT_POS_PLANE as usize;
    #[unroll]
    for r in 0..queries {
        if r < slot_box.seq_q {
            let merged = MergedSlotRow::<EA, N>::new(
                slot_out,
                slot_max,
                slot_sum,
                slot_box.first_row + r,
                slots,
            )
            .normalized();
            let row_start = slot_box.batch * out.stride(0)
                + slot_box.kv * out.stride(1)
                + slot_box.head * out.stride(2)
                + (slot_box.first_query + r) * out.stride(3);
            let element_stride = out.stride(4);
            #[unroll]
            for e in 0..elements {
                let line = lane + e * units;
                if line < lines {
                    #[unroll]
                    for j in 0..width {
                        out[row_start + (line * width + j) * element_stride] =
                            E::cast_from(merged[e].extract(j));
                    }
                }
            }
        }
    }
}

/// One plane's share of the rows a group of planes split the keys of, merged from the slots
/// they left ([`MergedSlotRow::new`]) and written into `out`, normalized: of the group's first
/// `queries` rows, which lie from row `first_row` of the slots as `slots` states, the plane
/// merges each `r` whose remainder by the planes splitting the keys is `split`, its place among
/// them. `out` is windowed to the group's rows. A row whose every slot saw nothing writes zeros.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plane_merge_rows_microkernel<EA: Float, E: Float, N: Size>(
    // Read only, taken as the cube holds them: cube code passes no `&mut` where a `&` is asked.
    slot_out: &mut [Vector<EA, N>],
    slot_max: &mut [EA],
    slot_sum: &mut [EA],
    out: &Tile,
    first_row: usize,
    split: usize,
    #[comptime] slots: AttentionSlotShape,
    #[comptime] queries: usize,
) {
    let AttentionSlotShape {
        units,
        val_dim,
        splits,
    } = slots;
    let width = N::value();
    let lines_of_value = comptime!(val_dim / width);
    let lane = UNIT_POS_PLANE as usize;
    let mut out = out.clone();
    let mut lines = out.flat_mut::<E, N>();
    #[unroll]
    for r in 0..queries {
        if r % splits == split {
            let merged =
                MergedSlotRow::<EA, N>::new(slot_out, slot_max, slot_sum, first_row + r, slots)
                    .normalized();
            #[unroll]
            for e in 0..comptime!(lines_of_value.div_ceil(units)) {
                let line = lane + e * units;
                if line < lines_of_value {
                    lines.write(r * lines_of_value + line, Vector::cast_from(merged[e]));
                }
            }
        }
    }
}

/// [`plane_merge_rows_microkernel`], its merged rows left undivided in slots rather than written:
/// of the first `rows`, row `r`'s line `l` at slot `first_slot + r · row_stride` times the lines
/// of a value, plus `l`, of `out`, its merged max and sum at that slot of `max` and `sum`. What a
/// merge across cubes reads is one slot a cube, whatever its planes.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plane_merge_rows_partial_microkernel<EA: Float, N: Size>(
    // Read only, taken as the cube holds them: cube code passes no `&mut` where a `&` is asked.
    slot_out: &mut [Vector<EA, N>],
    slot_max: &mut [EA],
    slot_sum: &mut [EA],
    out: &mut [Vector<EA, N>],
    max: &mut [EA],
    sum: &mut [EA],
    first_row: usize,
    split: usize,
    first_slot: usize,
    row_stride: usize,
    rows: usize,
    #[comptime] slots: AttentionSlotShape,
    #[comptime] queries: usize,
) {
    let AttentionSlotShape {
        units,
        val_dim,
        splits,
    } = slots;
    let width = N::value();
    let lines_of_value = comptime!(val_dim / width);
    let lane = UNIT_POS_PLANE as usize;
    #[unroll]
    for r in 0..queries {
        if r % splits == split && r < rows {
            let merged =
                MergedSlotRow::<EA, N>::new(slot_out, slot_max, slot_sum, first_row + r, slots);
            let slot = first_slot + r * row_stride;
            #[unroll]
            for e in 0..comptime!(lines_of_value.div_ceil(units)) {
                let line = lane + e * units;
                if line < lines_of_value {
                    out[slot * lines_of_value + line] = merged.lines[e];
                }
            }
            if lane == 0 {
                max[slot] = merged.max;
                sum[slot] = merged.sum;
            }
        }
    }
}

/// One row's slots merged into one partial, as a plane holds it: each unit's lines of the row's
/// unnormalized output, line `e` the row's line `lane + e · units`, and the row's merged max and
/// sum, the same on every unit. Itself a partial of the row where the slots held only some of its
/// keys, and the row's output once [`normalized`](Self::normalized).
#[derive(CubeType)]
struct MergedSlotRow<EA: Float, N: Size> {
    lines: Array<Vector<EA, N>>,
    max: EA,
    sum: EA,
}

#[cube]
impl<EA: Float, N: Size> MergedSlotRow<EA, N> {
    /// Row `row`'s slots merged by one plane, sharing the row's work: every unit reads every
    /// `units`-th slot's max to find the row's merged max and weighs its slots by `exp(max − merged
    /// max)`, the plane sums the weighed running sums, and each unit adds slots' outputs into its
    /// lines of the row, weighed by the weight the unit holding that slot hands round. Every weight
    /// is handed round before any output is read, so the reads of a row's slots are in flight
    /// together. A row whose every slot saw nothing merges to a sum of zero.
    ///
    /// A row of fewer lines than the plane has units would leave the units past them idle while the
    /// rest read every slot: where its lines are a power of two, the plane's units form groups of
    /// that many, each group reading every `groups`-th slot, and the groups' sums meet once every slot
    /// is read. A unit past the row's lines then holds its group's copy of line `lane % lines`.
    ///
    /// The slots lie in any memory, laid out as [`AttentionSlotShape`] states: device memory where
    /// cubes meet, shared memory where a cube's planes do.
    fn new(
        // Read only, taken as the kernel holds them: cube code passes no `&mut` where a `&` is asked.
        slot_out: &mut [Vector<EA, N>],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        row: usize,
        #[comptime] shape: AttentionSlotShape,
    ) -> MergedSlotRow<EA, N> {
        let AttentionSlotShape {
            units,
            val_dim,
            splits,
        } = shape;
        let width = slot_out.vector_size();
        let lines = comptime!(val_dim / width);
        let groups = comptime!(match lines < units && lines.is_power_of_two() {
            true => units / lines,
            false => 1,
        });
        let elements = comptime!(lines.div_ceil(units));
        let rounds = comptime!(splits.div_ceil(units));
        let per_group = comptime!(units.min(splits).div_ceil(groups));
        let lane = UNIT_POS_PLANE as usize;
        let (group, line_lane) = match comptime!(groups) {
            1 => (0usize, lane),
            _ => (lane / lines, lane % lines),
        };
        let first_slot = row * splits;
        let mut own_max = EA::min_value();
        #[unroll]
        for round in 0..rounds {
            let own = round * units + lane;
            if own < splits {
                own_max = max(own_max, slot_max[first_slot + own]);
            }
        }
        let merged = plane_max(own_max);
        let mut sums = Array::<Vector<EA, N>>::new(elements);
        #[unroll]
        for e in 0..elements {
            sums[e] = Vector::cast_from(0u32);
        }
        let mut total = EA::from_int(0);
        #[unroll]
        for round in 0..rounds {
            let own = round * units + lane;
            let mut weight = EA::from_int(0);
            if own < splits {
                weight = (slot_max[first_slot + own] - merged).exp();
                total += weight * slot_sum[first_slot + own];
            }
            // The weight of the `k`-th slot this unit's group reads in the round, from the unit holding it.
            let mut handed = Array::<EA>::new(per_group);
            #[unroll]
            for k in 0..per_group {
                handed[k] = match comptime!(groups) {
                    1 => plane_broadcast(weight, k as u32),
                    _ => plane_shuffle(weight, (k * groups + group) as u32),
                };
            }
            #[unroll]
            for k in 0..per_group {
                let index = round * units + k * groups + group;
                if index < splits {
                    let slot = first_slot + index;
                    #[unroll]
                    for e in 0..elements {
                        let line = line_lane + e * units;
                        if line < lines {
                            sums[e] += Vector::cast_from(handed[k]) * slot_out[slot * lines + line];
                        }
                    }
                }
            }
        }
        if comptime!(groups > 1) {
            let across_groups = comptime!(UnitShare::Group {
                unit_bits: (units - 1) & !(lines - 1)
            });
            sums[0] = across_groups.reduce::<Vector<EA, N>>(sums[0], Monoid::Sum);
        }
        MergedSlotRow::<EA, N> {
            lines: sums,
            max: merged,
            sum: plane_sum(total),
        }
    }

    /// The unit's lines divided by the row's sum: the row's output, zero where no slot saw a key.
    fn normalized(&mut self) -> Array<Vector<EA, N>> {
        let recip = select(
            self.sum > EA::from_int(0),
            EA::from_int(1) / self.sum,
            EA::from_int(0),
        );
        #[unroll]
        for e in 0..self.lines.len() {
            self.lines[e] *= Vector::cast_from(recip);
        }
        self.lines
    }
}
