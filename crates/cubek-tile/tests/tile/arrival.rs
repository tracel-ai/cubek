//! The last cube in merges what the others left, for a merge no atomic add can spell.
//!
//! The shape under test is attention's: every cube reduces a slice of one row to a running max
//! and the sum of the exponentials it implies. Two of those combine only after both are rescaled
//! to a common max; adding the partial sums straight is wrong, so this tests [`Arrival`].
//!
//! Two rows run at once, each with a counter of its own, because that is how a real launch is
//! shaped: the cubes fall into independent groups and every group's last cube merges its own.
//! An `Arrival` that counted the grid instead of the group would leave one row unwritten.

use cubecl::prelude::*;
use cubek_tile::*;

/// Cubes per row, units per cube, and the rows that run side by side.
const CUBES: u32 = 16;
const UNITS: u32 = 8;
const ROWS: u32 = 2;
/// The values one row holds.
const ROW: usize = (CUBES * UNITS) as usize;

/// Every unit takes one value of the row. Its cube reduces them to `(max, sum)`, publishes that
/// pair, and announces itself; the cube that is last rescales every pair to the greatest max and
/// adds the sums, which is the denominator of the row's softmax.
#[cube(launch)]
fn softmax_denominator(
    values: &[f32],
    maxes: &mut [f32],
    sums: &mut [f32],
    counter: &mut [Atomic<u32>],
    out: &mut [f32],
    #[comptime] cubes: u32,
    #[comptime] units: u32,
) {
    // The grid is `(split, row)`: x deals a row's cubes, y the rows, so each row's cubes share
    // a counter and nothing else.
    let row = CUBE_POS_Y;
    let split = CUBE_POS_X;
    let mine = split * units + UNIT_POS;
    let value = values[(row * comptime!(CUBES * UNITS) + mine) as usize];

    // The cube's own reduction, which is ordinary shared-memory work.
    let mut cube_values = Shared::<[f32]>::new_slice(units as usize);
    let mut cube_max = Shared::<f32>::new();
    cube_values[UNIT_POS as usize] = value;
    sync_cube();
    if UNIT_POS == 0 {
        let mut largest = cube_values[0];
        let mut i = 1u32;
        while i < units {
            largest = max(largest, cube_values[i as usize]);
            i += 1u32;
        }
        *cube_max = largest;
    }
    sync_cube();

    let cube_largest = *cube_max;
    cube_values[UNIT_POS as usize] = Exp::exp(value - cube_largest);
    sync_cube();
    if UNIT_POS == 0 {
        let mut sum = 0f32;
        let mut i = 0u32;
        while i < units {
            sum += cube_values[i as usize];
            i += 1u32;
        }
        maxes[(row * cubes + split) as usize] = cube_largest;
        sums[(row * cubes + split) as usize] = sum;
    }

    // Every unit reaches it: the release covers what unit zero just published, and the count is
    // taken once and read by the whole cube.
    if comptime!(Arrival::new(cubes)).count_in(&counter[row as usize]) && UNIT_POS == 0 {
        let base = row * cubes;
        let mut merged_max = maxes[base as usize];
        let mut c = 1u32;
        while c < cubes {
            merged_max = max(merged_max, maxes[(base + c) as usize]);
            c += 1u32;
        }
        // The rescale is the whole point: a sum taken against one max only counts against
        // another once it is brought onto it.
        let mut merged_sum = 0f32;
        let mut c = 0u32;
        while c < cubes {
            merged_sum +=
                sums[(base + c) as usize] * Exp::exp(maxes[(base + c) as usize] - merged_max);
            c += 1u32;
        }
        out[(row * 2) as usize] = merged_sum;
        out[(row * 2 + 1) as usize] = merged_max;
    }
}

/// Row `row`, spread wide enough that a slice's own max is far from the row's: rescaling a
/// partial taken against a max 30 below it is what the merge has to get right. The rows differ,
/// so a merge that read the wrong group's partials answers wrongly rather than by luck.
fn row(row: u32) -> Vec<f32> {
    (0..ROW)
        .map(|i| i as f32 * 0.25 + row as f32 * 7.0)
        .collect()
}

/// A row's softmax denominator and its max, computed straight.
fn expected(row: u32) -> (f32, f32) {
    let values = self::row(row);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (values.iter().map(|v| (v - max).exp()).sum(), max)
}

#[test]
fn the_last_cube_in_merges_the_others() {
    let client = cubecl::test_device().client();
    if !client.properties().features.device_memory_scope {
        // Without the scope a cube may read another's partial stale, so there is nothing to
        // test here; the device wants the second dispatch instead.
        println!("device memory scope not supported - skipped");
        return;
    }

    let all: Vec<f32> = (0..ROWS).flat_map(row).collect();
    let slots = (ROWS * CUBES) as usize;
    let values = client.create_from_slice(f32::as_bytes(&all));
    let maxes = client.empty(slots * core::mem::size_of::<f32>());
    let sums = client.empty(slots * core::mem::size_of::<f32>());
    let counter = client.create_from_slice(u32::as_bytes(&vec![0u32; ROWS as usize]));
    let out = client.create_from_slice(f32::as_bytes(&vec![0f32; 2 * ROWS as usize]));

    softmax_denominator::launch(
        &client,
        CubeCount::Static(CUBES, ROWS, 1),
        CubeDim::new_1d(UNITS),
        unsafe { BufferArg::from_raw_parts(values, all.len()) },
        unsafe { BufferArg::from_raw_parts(maxes, slots) },
        unsafe { BufferArg::from_raw_parts(sums, slots) },
        unsafe { BufferArg::from_raw_parts(counter, ROWS as usize) },
        unsafe { BufferArg::from_raw_parts(out.clone(), 2 * ROWS as usize) },
        CUBES,
        UNITS,
    );

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);
    for row in 0..ROWS as usize {
        let (sum, max) = expected(row as u32);
        assert!(
            (actual[row * 2] - sum).abs() < sum * 1e-4,
            "row {row} denominator: {} against {sum}",
            actual[row * 2]
        );
        assert_eq!(actual[row * 2 + 1], max, "row {row} max");
    }
}
