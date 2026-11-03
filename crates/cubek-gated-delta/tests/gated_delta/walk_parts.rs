//! The parts the walk's microkernel is composed of, each against a host twin, at shapes chosen to
//! be awkward: lines narrower than the run's columns, and key rows the last units overhang.
//!
//! A probe kernel runs one part and dumps what it holds.

use super::harness::{F32, read};
use super::walk::{Case, assert_close};
use cubecl::bytes::Bytes;
use cubecl::tile::{Partitioning, Region, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_gated_delta::kernels::walk::{
    CubeGatedDeltaWalkRoutine, CubeGatedDeltaWalkRoutineSettings, GatedDeltaWalkTensors,
};
use cubek_microkernel::gated_delta::walk::cube::{
    CubeGatedDeltaWalkMicrokernel, HeldState, UnitPartials,
};
use cubek_microkernel::gated_delta::{GatedDeltaHeads, KEY_FEATURE, POSITION, VALUE_FEATURE};
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// What unit `unit` of run `run` puts in column `column` for sum `sum` at `position`.
fn partial_of(unit: usize, run: usize, position: usize, sum: usize, column: usize) -> f32 {
    0.5 * (unit * 7 + column + run * 3 + position) as f32 + 1.0 - 0.75 * (sum * (column + 1)) as f32
}

/// Every unit of every run fills its partials as the twin above says, meets in both sums at each
/// of `positions`, and leaves what it holds in `dump`: the first sum's columns, then the second's.
#[cube(launch_unchecked)]
fn unit_partials_probe<N: Size>(
    dump: &mut [f32],
    #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
    #[comptime] runs: usize,
    positions: u32,
) {
    let width = N::value();
    let columns = comptime!(microkernel.columns);
    let parts = comptime!(columns / width);
    let units = comptime!(microkernel.units);
    let (unit, run) = (UNIT_POS_X as usize, UNIT_POS_Y as usize);
    let mut partials = UnitPartials::<N>::new(run, unit, runs, microkernel);
    for position in 0..positions as usize {
        let base = ((position * runs + run) * units + unit) * comptime!(2 * columns);
        #[unroll]
        for sum in 0..2usize {
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
            partials.sum(sum);
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

/// Unit counts of a power of two, at lines as wide as the run's columns and narrower, meeting in both sums over several positions: each unit holds the sum of its run's
/// partials, and a sum never reads another's.
#[test]
fn unit_partials_sum_like_the_host() {
    let client = cubecl::test_device().client();
    let hardware = Hardware::from_client(&client).properties.hardware.clone();
    let runs = 2usize;
    let positions = 3usize;
    for units in [1usize, 2, 4, 8, 16] {
        if units * runs > hardware.max_units_per_cube as usize
            || units > hardware.max_cube_dim.0 as usize
            || runs > hardware.max_cube_dim.1 as usize
        {
            eprintln!("skipped, the device holds no cube of {units} units by {runs} runs");
            continue;
        }
        for (columns, width) in [(1usize, 1usize), (2, 2), (4, 4), (4, 2), (4, 1)] {
            let microkernel = CubeGatedDeltaWalkMicrokernel {
                units,
                rows: columns,
                columns,
            };
            let stride = 2 * columns;
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
                    microkernel,
                    runs,
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
                                 position {position} run {run} unit {unit}"
                            ),
                        );
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
/// position in `dump`, write a fixed delta, and keep the states.
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
    #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
    #[comptime] runs: usize,
) {
    let width = N::value();
    let columns = comptime!(microkernel.columns);
    let parts = comptime!(columns / width);
    let units = comptime!(microkernel.units);
    let mut write = Array::<Vector<f32, N>>::new(parts);
    for run in cube {
        for unit in run.walk() {
            let slot = unit.coord(KEY_FEATURE);
            let mut partials =
                UnitPartials::<N>::new(run.coord(VALUE_FEATURE), slot, runs, microkernel);
            let mut held = HeldState::<N>::load(&state.at(&unit), microkernel);
            let origin = unit.origin(VALUE_FEATURE);
            for step in unit.walk() {
                let position = step.coord(POSITION);
                let (keys, queries) = (key.at(&step), query.at(&step));
                let base = (origin / columns * units + slot) * comptime!(2 * columns);
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
    #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
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
            microkernel,
            runs,
        );
    }
}

/// Heads whose key rows the last units overhang, a prime head, and a wide one: every unit's partials of
/// the first position, before and after the write, and the states the walk keeps, against the
/// host.
#[test]
fn held_state_recalls_answers_and_writes_like_the_host() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for (key_features, value_features) in [(8, 4), (40, 36), (37, 53), (100, 12), (128, 128)] {
        let case = Case {
            batch: 1,
            positions: 3,
            heads: GatedDeltaHeads {
                key_heads: 1,
                group: 1,
                key_features,
                value_features,
            },
            kept: 3,
            masked: vec![],
        };
        for settings in [
            CubeGatedDeltaWalkRoutineSettings::FLOOR,
            CubeGatedDeltaWalkRoutineSettings::DEFAULT,
        ] {
            let o = case.operands();
            let tensors = GatedDeltaWalkTensors::new(case.inputs(&client, &o, F32)).unwrap();
            let prepared = CubeGatedDeltaWalkRoutine.prepare(&hw, tensors.problem(), &settings);
            let (blueprint, grid) = match prepared {
                Ok(prepared) => prepared,
                Err(why) => {
                    eprintln!("skipped, the device declines {settings} {case:?}: {why:?}");
                    assert_eq!(tensors.problem().admit(&hw), Err(why));
                    continue;
                }
            };
            let microkernel = blueprint.microkernel;
            let columns = microkernel.columns;
            let runs = value_features / columns;
            let stride = 2 * columns;
            let len = runs * microkernel.units * stride;
            let dump = client.create(Bytes::from_elems(vec![f32::NAN; len]));
            let binding = tensors.bind(&blueprint, &grid);
            let operands = binding.operands;
            // SAFETY: every operand is bound through the launcher as the real kernel's are, and
            // each unit writes its own `stride` floats of `dump`, within `len`.
            unsafe {
                held_state_probe::launch_unchecked(
                    tensors.client(),
                    grid.launch.cube_count.clone(),
                    grid.launch.cube_dim,
                    blueprint.width,
                    operands.query,
                    operands.key,
                    operands.state,
                    operands.kept,
                    BufferArg::from_raw_parts(dump.clone(), len),
                    tensors.problem().first_kept(),
                    binding.launcher.partitioning_arg(),
                    microkernel,
                    blueprint.runs,
                );
            }
            let what = format!("{settings} {microkernel} {case:?}");
            let got = read(&client, dump, F32);
            let (k0, q0) = (&o.key[..key_features], &o.query[..key_features]);
            let state = |r: usize, column: usize| o.state[r * value_features + column] as f64;
            for run in 0..runs {
                for unit in 0..microkernel.units {
                    let rows =
                        unit * microkernel.rows..((unit + 1) * microkernel.rows).min(key_features);
                    let mut expected = vec![0.0f64; stride];
                    for c in 0..columns {
                        let column = run * columns + c;
                        for r in rows.clone() {
                            let decayed = state(r, column) * PROBE_DECAY as f64;
                            expected[c] += decayed * k0[r] as f64;
                            let written = decayed + k0[r] as f64 * probe_delta(column);
                            expected[columns + c] += written * q0[r] as f64;
                        }
                    }
                    let base = (run * microkernel.units + unit) * stride;
                    assert_close(
                        got[base..base + stride].to_vec(),
                        &expected,
                        &format!("{what}: run {run} unit {unit}"),
                    );
                }
            }
            // The states after each position.
            let walked = tensors.into_outputs();
            let kept = read(&client, walked.kept.handle.clone(), F32);
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
