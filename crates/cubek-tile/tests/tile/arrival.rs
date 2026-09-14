//! The last cube in merges what the others left, for a merge no atomic add can spell.
//!
//! The shape under test is attention's: every cube reduces a slice of one row to a running max
//! and the sum of the exponentials it implies, and combining two of those is not addition — the
//! sums are on different scales until a common max is chosen and both are rescaled to it. Adding
//! the partial sums straight gives a different, wrong answer, which is what makes this a test of
//! [`last_cube_in`] rather than of the drain beside it.

use cubecl::prelude::*;
use cubek_tile::*;

/// Cubes, and units per cube. The row is their product.
const CUBES: u32 = 16;
const UNITS: u32 = 8;
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
    let mine = CUBE_POS as u32 * units + UNIT_POS;
    let value = values[mine as usize];

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
        maxes[CUBE_POS] = cube_largest;
        sums[CUBE_POS] = sum;
    }

    // Every unit reaches it: the release covers what unit zero just published, and the count is
    // taken once and read by the whole cube.
    if last_cube_in(&counter[0]) && UNIT_POS == 0 {
        let mut merged_max = maxes[0];
        let mut c = 1u32;
        while c < cubes {
            merged_max = max(merged_max, maxes[c as usize]);
            c += 1u32;
        }
        // The rescale is the whole point: a sum taken against one max only counts against
        // another once it is brought onto it.
        let mut merged_sum = 0f32;
        let mut c = 0u32;
        while c < cubes {
            merged_sum += sums[c as usize] * Exp::exp(maxes[c as usize] - merged_max);
            c += 1u32;
        }
        out[0] = merged_sum;
        out[1] = merged_max;
    }
}

/// The row, spread wide enough that a slice's own max is far from the row's: rescaling a partial
/// that was taken against a max 30 below the row's is what the merge has to get right.
fn row() -> Vec<f32> {
    (0..ROW).map(|i| i as f32 * 0.25).collect()
}

/// The denominator and the max, computed straight.
fn expected() -> (f32, f32) {
    let row = row();
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (row.iter().map(|v| (v - max).exp()).sum(), max)
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

    let values = client.create_from_slice(f32::as_bytes(&row()));
    let maxes = client.empty(CUBES as usize * core::mem::size_of::<f32>());
    let sums = client.empty(CUBES as usize * core::mem::size_of::<f32>());
    let counter = client.create_from_slice(u32::as_bytes(&[0u32]));
    let out = client.create_from_slice(f32::as_bytes(&[0f32; 2]));

    softmax_denominator::launch(
        &client,
        CubeCount::Static(CUBES, 1, 1),
        CubeDim::new_1d(UNITS),
        unsafe { BufferArg::from_raw_parts(values, ROW) },
        unsafe { BufferArg::from_raw_parts(maxes, CUBES as usize) },
        unsafe { BufferArg::from_raw_parts(sums, CUBES as usize) },
        unsafe { BufferArg::from_raw_parts(counter, 1) },
        unsafe { BufferArg::from_raw_parts(out.clone(), 2) },
        CUBES,
        UNITS,
    );

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);
    let (sum, max) = expected();

    assert!(
        (actual[0] - sum).abs() < sum * 1e-4,
        "denominator: {} against {sum}",
        actual[0]
    );
    assert_eq!(actual[1], max);
}
