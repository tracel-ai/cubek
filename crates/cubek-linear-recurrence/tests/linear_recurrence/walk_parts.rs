//! The parts the walk's microkernel is composed of, each against a host twin, at shapes chosen to
//! be awkward: lines narrower than the run's columns, and key rows the last units overhang.
//!
//! A probe kernel runs one part and dumps what it holds.

use super::harness::{F32, read};
use super::walk::{Case, assert_close};
use cubecl::bytes::Bytes;
use cubecl::tile::{Partitioning, Region, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_linear_recurrence::kernels::walk::{
    CubeDeltaWalkRoutine, CubeWalkRoutineSettings, RecurrenceWalkTensors,
};
use cubek_microkernel::linear_recurrence::walk::UpdateRule;
use cubek_microkernel::linear_recurrence::walk::cube::{
    CubeWalkMicrokernelSettings, HeldState, HeldStateShape, UnitPartials,
};
use cubek_microkernel::linear_recurrence::{KEY_FEATURE, POSITION, RecurrenceHeads, VALUE_FEATURE};
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// What unit `unit` of run `run` puts in column `column` for sum `sum` at `position`.
fn partial_of(unit: usize, run: usize, position: usize, sum: usize, column: usize) -> f32 {
    0.5 * (unit * 7 + column + run * 3 + position) as f32 + 1.0 - 0.75 * (sum * (column + 1)) as f32
}

/// Every unit of every run fills its partials as the twin above says, meets in `sums_a_position`
/// sums at each of `positions`, and leaves what it holds in `dump`: the first sum's columns, then
/// the second's.
#[cube(launch_unchecked)]
fn unit_partials_probe<N: Size>(
    dump: &mut [f32],
    #[comptime] shape: HeldStateShape,
    #[comptime] runs: usize,
    #[comptime] sums_a_position: usize,
    positions: u32,
) {
    let width = N::value();
    let columns = comptime!(shape.columns());
    let parts = comptime!(columns / width);
    let units = comptime!(shape.units());
    let (unit, run) = (UNIT_POS_X as usize, UNIT_POS_Y as usize);
    let mut partials = UnitPartials::<N>::new(run, unit, runs, shape);
    for position in 0..positions as usize {
        let base = ((position * runs + run) * units + unit) * comptime!(sums_a_position * columns);
        #[unroll]
        for sum in 0..sums_a_position {
            partials.clear();
            #[unroll]
            for part in 0..parts {
                let mut line = Vector::<f32, N>::cast_from(0.0f32);
                #[unroll]
                for lane in 0..width {
                    let column = part * width + lane;
                    line.insert(
                        lane,
                        0.5f32 * f32::cast_from(unit * 7 + column + run * 3 + position) + 1.0f32
                            - 0.75f32 * f32::cast_from(sum * (column + 1)),
                    );
                }
                partials.add(part, line);
            }
            partials.sum();
            #[unroll]
            for part in 0..parts {
                #[unroll]
                for lane in 0..width {
                    dump[base + sum * columns + part * width + lane] =
                        partials.get(part).extract(lane);
                }
            }
        }
    }
}

/// Unit counts of a power of two, at lines as wide as the run's columns and narrower, meeting in
/// one sum and in two a position over many positions: each unit holds the sum of its run's
/// partials, and a sum never reads another's.
#[test]
fn unit_partials_sum_like_the_host() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let hardware = hw.properties.hardware.clone();
    let runs = 2usize;
    let positions = 17usize;
    for units in [1usize, 2, 4, 8, 16] {
        if units * runs > hardware.max_units_per_cube as usize
            || units > hardware.max_cube_dim.0 as usize
            || runs > hardware.max_cube_dim.1 as usize
        {
            eprintln!("skipped, the device holds no cube of {units} units by {runs} runs");
            continue;
        }
        for (columns, width) in [(1usize, 1usize), (2, 2), (4, 4), (4, 2), (4, 1)] {
            for sums_a_position in [1usize, 2] {
                let heads = RecurrenceHeads {
                    key_heads: 1,
                    heads_per_group: 1,
                    key_features: units * columns,
                    value_features: columns,
                };
                let shape =
                    HeldStateShape::new(&hw, &heads, &CubeWalkMicrokernelSettings::new(columns))
                        .expect("the device holds the units the probe skips none of");
                let stride = sums_a_position * columns;
                let len = positions * runs * units * stride;
                let dump = client.create(Bytes::from_elems(vec![f32::NAN; len]));
                // SAFETY: each unit writes its own `stride` floats of `dump`, within `len`.
                unsafe {
                    unit_partials_probe::launch_unchecked(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_2d(units as u32, runs as u32),
                        width,
                        BufferArg::from_raw_parts(dump.clone(), len),
                        shape,
                        runs,
                        sums_a_position,
                        positions as u32,
                    );
                }
                let got = read(&client, dump, F32);
                for position in 0..positions {
                    for run in 0..runs {
                        let expected: Vec<f64> = (0..stride)
                            .map(|at| {
                                let (sum, column) = (at / columns, at % columns);
                                (0..units)
                                    .map(|u| partial_of(u, run, position, sum, column) as f64)
                                    .sum()
                            })
                            .collect();
                        for unit in 0..units {
                            let base = ((position * runs + run) * units + unit) * stride;
                            assert_close(
                                got[base..base + stride].to_vec(),
                                &expected,
                                &format!(
                                    "units={units} columns={columns} width={width} \
                                     sums={sums_a_position} position {position} run {run} \
                                     unit {unit}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The decay a probe applies and the delta it adds along each key.
const PROBE_DECAY: f32 = 0.9;

fn probe_delta(column: usize) -> f64 {
    0.1 * (column + 3) as f64
}

/// One cube's runs load their state, leave each unit's recall and answer partials of the first
/// position in `dump`, write a fixed delta, and keep the states; a second load of the state then
/// decays and writes that delta in one pass, and leaves its answer.
#[cube]
#[allow(clippy::too_many_arguments)]
fn held_state_part<N: Size>(
    cube: &Region,
    query: &Tile,
    key: &Tile,
    state: &Tile,
    kept: &Tile,
    dump: &mut [f32],
    first_kept: usize,
    #[comptime] shape: HeldStateShape,
    #[comptime] runs: usize,
) {
    let width = N::value();
    let columns = comptime!(shape.columns());
    let parts = comptime!(columns / width);
    let units = comptime!(shape.units());
    let mut write = Array::<Vector<f32, N>>::new(parts);
    for run in cube {
        for unit in run.walk() {
            let slot = unit.coord(KEY_FEATURE);
            let mut partials = UnitPartials::<N>::new(run.coord(VALUE_FEATURE), slot, runs, shape);
            let mut held = HeldState::<N>::load(&state.at(&unit), false, shape);
            let origin = unit.origin(VALUE_FEATURE);
            for step in unit.walk() {
                let position = step.coord(POSITION);
                let (keys, queries) = (key.at(&step), query.at(&step));
                let base = (origin / columns * units + slot) * comptime!(3 * columns);
                held.decay_and_recall(&keys, PROBE_DECAY, &mut partials);
                if position == 0 {
                    #[unroll]
                    for part in 0..parts {
                        #[unroll]
                        for lane in 0..width {
                            dump[base + part * width + lane] = partials.get(part).extract(lane);
                        }
                    }
                }
                #[unroll]
                for part in 0..parts {
                    let mut line = Vector::<f32, N>::cast_from(0.0f32);
                    #[unroll]
                    for lane in 0..width {
                        line.insert(
                            lane,
                            0.1f32 * f32::cast_from(origin + part * width + lane + 3),
                        );
                    }
                    write[part] = line;
                }
                held.write_and_answer(&keys, &queries, &write, &mut partials);
                if position == 0 {
                    #[unroll]
                    for part in 0..parts {
                        #[unroll]
                        for lane in 0..width {
                            dump[base + columns + part * width + lane] =
                                partials.get(part).extract(lane);
                        }
                    }
                    let mut fused = HeldState::<N>::load(&state.at(&unit), false, shape);
                    fused.decay_write_and_answer(
                        &keys,
                        &queries,
                        PROBE_DECAY,
                        &write,
                        &mut partials,
                    );
                    #[unroll]
                    for part in 0..parts {
                        #[unroll]
                        for lane in 0..width {
                            dump[base + 2 * columns + part * width + lane] =
                                partials.get(part).extract(lane);
                        }
                    }
                }
                if position >= first_kept {
                    held.store(&mut kept.at(&unit), position - first_kept);
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn held_state_probe<N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    state: &TileTensor<'_>,
    kept: &TileTensor<'_>,
    dump: &mut [f32],
    first_kept: usize,
    partitioning: Partitioning,
    #[comptime] shape: HeldStateShape,
    #[comptime] runs: usize,
) {
    let query = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let state = state.tile(&partitioning);
    let kept = kept.tile(&partitioning);
    for cube in &partitioning {
        held_state_part::<N>(
            &cube,
            &query.at(&cube),
            &key.at(&cube),
            &state.at(&cube),
            &kept.at(&cube),
            dump,
            first_kept,
            shape,
            runs,
        );
    }
}

/// Heads whose key rows the last units overhang, a prime head, and a wide one: every unit's
/// partials of the first position, before and after the write and after the fused decay and
/// write, and the states the walk keeps, against the host.
#[test]
fn held_state_recalls_answers_and_writes_like_the_host() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for (key_features, value_features) in [(8, 4), (40, 36), (37, 53), (100, 12), (128, 128)] {
        let case = Case {
            batch: 1,
            positions: 3,
            heads: RecurrenceHeads {
                key_heads: 1,
                heads_per_group: 1,
                key_features,
                value_features,
            },
            kept: 3,
            rule: UpdateRule::Delta,
            masked: vec![],
        };
        for settings in [
            CubeWalkRoutineSettings::FLOOR,
            CubeWalkRoutineSettings::DEFAULT,
        ] {
            let o = case.operands();
            let tensors = RecurrenceWalkTensors::new(case.inputs(&client, &o, F32)).unwrap();
            let prepared = CubeDeltaWalkRoutine.prepare(&hw, tensors.problem(), &settings);
            let (blueprint, grid) = match prepared {
                Ok(prepared) => prepared,
                Err(why) => {
                    eprintln!("skipped, the device declines {settings} {case:?}: {why:?}");
                    assert_eq!(tensors.problem().admit(&hw), Err(why));
                    continue;
                }
            };
            let shape = blueprint.microkernel.shape;
            let columns = shape.columns();
            let runs = value_features / columns;
            let stride = 3 * columns;
            let len = runs * shape.units() * stride;
            let dump = client.create(Bytes::from_elems(vec![f32::NAN; len]));
            let binding = tensors.bind(&blueprint, &grid);
            let kept = binding.kept.expect("the case owns its states");
            // SAFETY: every operand is bound through the launcher as the real kernel's are, and
            // each unit writes its own `stride` floats of `dump`, within `len`.
            unsafe {
                held_state_probe::launch_unchecked(
                    tensors.client(),
                    grid.launch.cube_count.clone(),
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.query,
                    binding.key,
                    binding.state,
                    kept,
                    BufferArg::from_raw_parts(dump.clone(), len),
                    tensors.problem().first_kept(),
                    binding.launcher.partitioning_arg(),
                    shape,
                    blueprint.runs,
                );
            }
            let what = format!("{settings} {shape} {case:?}");
            let got = read(&client, dump, F32);
            let (k0, q0) = (&o.key[..key_features], &o.query[..key_features]);
            let state = |r: usize, column: usize| o.state[r * value_features + column] as f64;
            for run in 0..runs {
                for unit in 0..shape.units() {
                    let rows = unit * shape.rows()..((unit + 1) * shape.rows()).min(key_features);
                    let mut expected = vec![0.0f64; stride];
                    for c in 0..columns {
                        let column = run * columns + c;
                        for r in rows.clone() {
                            let decayed = state(r, column) * PROBE_DECAY as f64;
                            expected[c] += decayed * k0[r] as f64;
                            let written = decayed + k0[r] as f64 * probe_delta(column);
                            expected[columns + c] += written * q0[r] as f64;
                            expected[2 * columns + c] += written * q0[r] as f64;
                        }
                    }
                    let base = (run * shape.units() + unit) * stride;
                    assert_close(
                        got[base..base + stride].to_vec(),
                        &expected,
                        &format!("{what}: run {run} unit {unit}"),
                    );
                }
            }
            // The states after each position.
            let walked = tensors.into_outputs();
            let kept = read(&client, walked.kept.unwrap().handle.clone(), F32);
            let mut s: Vec<f64> = o.state.iter().map(|&x| x as f64).collect();
            let mut expected = Vec::new();
            for t in 0..case.positions {
                let k = &o.key[t * key_features..(t + 1) * key_features];
                for r in 0..key_features {
                    for j in 0..value_features {
                        let at = r * value_features + j;
                        s[at] = s[at] * PROBE_DECAY as f64 + k[r] as f64 * probe_delta(j);
                    }
                }
                expected.extend(&s);
            }
            assert_close(kept, &expected, &format!("{what}: the kept states"));
        }
    }
}
