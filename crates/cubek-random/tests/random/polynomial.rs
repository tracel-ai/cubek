use cubecl::{
    CubeDim, TestRuntime,
    prelude::{BufferArg, *},
};
use cubek_random::polynomial;

const POINTS: usize = 262_144;

/// The polynomial `cos` and `sin` land within the `1e-6` Box-Muller needs, everywhere on
/// the turn.
///
/// The quadrant comes from a truncating conversion and is undone by a swap and two sign
/// flips. A flip on the wrong quadrant only rotates the circle the transform draws from,
/// which every distribution test would still call normal.
#[test]
fn cos_sin_turns_match_the_host_library() {
    let turns: Vec<f32> = (0..POINTS).map(|i| i as f32 / POINTS as f32).collect();
    let (cosines, sines) = run_cos_sin_turns(&turns);

    let mut worst = 0.0f64;
    for ((turn, cosine), sine) in turns.iter().zip(&cosines).zip(&sines) {
        let angle = 2.0 * std::f64::consts::PI * *turn as f64;
        worst = worst.max((*cosine as f64 - angle.cos()).abs());
        worst = worst.max((*sine as f64 - angle.sin()).abs());
    }

    assert!(worst < 1e-6, "worst absolute error {worst}");
}

/// The polynomial `ln` stays within a relative `1e-6` over every normal input below one.
///
/// The Box-Muller radius is the square root of the logarithm, so an error there is a
/// stretched tail, which the 68-95-99 rule only notices once it is large.
#[test]
fn ln_matches_the_host_library() {
    let inputs = below_one_sweep();
    let logarithms = run_ln(&inputs);

    let mut worst = 0.0f64;
    for (input, logarithm) in inputs.iter().zip(&logarithms) {
        let expected = (*input as f64).ln();
        worst = worst.max((*logarithm as f64 - expected).abs() / expected.abs());
    }

    assert!(worst < 1e-6, "worst relative error {worst}");
}

/// Log-spaced over thirty decades, then over the last octave below one, where the
/// exponent contributes nothing and the series carries the result alone.
fn below_one_sweep() -> Vec<f32> {
    let half = POINTS / 2;
    let decades = (0..half).map(|i| 10f64.powf(-30.0 * (half - i) as f64 / half as f64) as f32);
    let last_octave =
        (0..half).map(|i| (1.0 - 0.5f64.powf(1.0 + 23.0 * i as f64 / half as f64)) as f32);

    decades.chain(last_octave).collect()
}

#[cube(launch)]
fn kernel_cos_sin_turns(turns: &[f32], cosines: &mut [f32], sines: &mut [f32]) {
    if ABSOLUTE_POS < turns.len() {
        let (cosine, sine) =
            polynomial::cos_sin_turns(Vector::<f32, Const<1>>::new(turns[ABSOLUTE_POS]));

        cosines[ABSOLUTE_POS] = cosine.extract(0usize);
        sines[ABSOLUTE_POS] = sine.extract(0usize);
    }
}

#[cube(launch)]
fn kernel_ln(inputs: &[f32], logarithms: &mut [f32]) {
    if ABSOLUTE_POS < inputs.len() {
        let logarithm = polynomial::ln(Vector::<f32, Const<1>>::new(inputs[ABSOLUTE_POS]));

        logarithms[ABSOLUTE_POS] = logarithm.extract(0usize);
    }
}

fn run_cos_sin_turns(turns: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let client = TestRuntime::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(turns));
    let cosines = client.empty(size_of_val(turns));
    let sines = client.empty(size_of_val(turns));

    let cube_dim = CubeDim::new(&client, turns.len());
    let cubes = turns.len().div_ceil(cube_dim.num_elems() as usize) as u32;

    kernel_cos_sin_turns::launch::<TestRuntime>(
        &client,
        CubeCount::Static(cubes, 1, 1),
        cube_dim,
        unsafe { BufferArg::from_raw_parts(input, turns.len()) },
        unsafe { BufferArg::from_raw_parts(cosines.clone(), turns.len()) },
        unsafe { BufferArg::from_raw_parts(sines.clone(), turns.len()) },
    );

    (
        f32::from_bytes(&client.read_one(cosines).unwrap()).to_vec(),
        f32::from_bytes(&client.read_one(sines).unwrap()).to_vec(),
    )
}

fn run_ln(inputs: &[f32]) -> Vec<f32> {
    let client = TestRuntime::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(inputs));
    let logarithms = client.empty(size_of_val(inputs));

    let cube_dim = CubeDim::new(&client, inputs.len());
    let cubes = inputs.len().div_ceil(cube_dim.num_elems() as usize) as u32;

    kernel_ln::launch::<TestRuntime>(
        &client,
        CubeCount::Static(cubes, 1, 1),
        cube_dim,
        unsafe { BufferArg::from_raw_parts(input, inputs.len()) },
        unsafe { BufferArg::from_raw_parts(logarithms.clone(), inputs.len()) },
    );

    f32::from_bytes(&client.read_one(logarithms).unwrap()).to_vec()
}
