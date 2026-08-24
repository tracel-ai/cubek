use cubecl::std::tensor::layout::{
    Coords1d,
    linear::{LinearViewMut, linear_view},
};
use cubecl::{prelude::*, std::tensor::ViewMut};
use cubecl_environment::{rand::get_seeded_rng, sync::Mutex};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{PrngBlueprint, PrngLaunchSettings, PrngState, PrngStrategy, Seeds, SeedsLaunch};

/// Values a unit produces under [`PrngBlueprint::Interleaved`].
pub(crate) const N_VALUES_PER_THREAD: usize = 128;

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Held across a "seed → derive seeds → submit random kernel" sequence so two
/// threads can't interleave their `seed()` and `get_seeds()` calls (which would
/// make each thread's launched kernel use the other thread's seeded state).
static SEED_GUARD: Mutex<()> = Mutex::new(());

pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    let mut seed = SEED.lock();
    *seed = Some(rng);
}

/// Install `seed` as the active RNG state and run `f` while holding a
/// process-wide guard. Use this whenever `f` calls one of the `random_*`
/// launchers: it keeps the seed-set and the kernel launch as a single
/// critical section, so parallel callers don't stomp on each other's seeds
/// between the two steps.
pub fn with_seed<R>(seed_value: u64, f: impl FnOnce() -> R) -> R {
    let _guard = SEED_GUARD.lock();
    seed(seed_value);
    f()
}

/// Pseudo-random generator
pub(crate) fn random<F: RandomFamily, R: Runtime>(
    client: &ComputeClient<R>,
    prng: F::Runtime,
    output: TensorBinding<R>,
    dtype: ElemType,
    strategy: PrngStrategy,
) -> Result<(), LaunchError> {
    let seeds = get_seeds();
    let args = prng.args();
    let launch = PrngLaunchSettings::new(
        client,
        &output,
        dtype,
        <F::Runtime as PrngArgs>::VECTORS_PER_DRAW,
        strategy,
    );

    let address_type = output.required_address_type(dtype.size());
    let output = linear_view(output);

    prng_kernel::launch::<F, R>(
        client,
        launch.cube_count,
        launch.cube_dim,
        address_type,
        launch.line_size,
        output,
        SeedsLaunch::new(seeds[0], seeds[1], seeds[2], seeds[3]),
        args,
        launch.vectors_per_unit,
        launch.blueprint,
        dtype,
    );

    Ok(())
}

pub(crate) fn get_seeds() -> [u32; 4] {
    let mut seed = SEED.lock();
    let mut rng: StdRng = match seed.take() {
        Some(rng_seeded) => rng_seeded,
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.random());
    }
    *seed = Some(rng);

    seeds.try_into().unwrap()
}

pub(crate) trait PrngArgs: Send + Sync + 'static {
    type Args: LaunchArg;

    /// Output vectors one draw writes: the Box-Muller transform emits a pair, the
    /// other distributions one each.
    const VECTORS_PER_DRAW: usize;

    fn args<R: Runtime>(self) -> <Self::Args as LaunchArg>::RuntimeArg<R>;
}

pub(crate) trait RandomFamily: Send + Sync + 'static + std::fmt::Debug {
    type Runtime: PrngRuntime;
}

#[cube]
pub(crate) trait PrngRuntime: Send + Sync + 'static + PrngArgs {
    /// The distribution's parameters, broadcast to the line.
    type Params<N: Size>: CubeType;

    /// Broadcast the parameters once, so a unit's draws share one set of splats.
    fn params<N: Size>(args: &Self::Args) -> Self::Params<N>;

    /// Turn the `nth` draw of a unit's state into its output vectors.
    ///
    /// The blueprint rides along because it names the device as well as the layout, and
    /// a distribution needing a transcendental pays a price on a CPU it does not on a GPU.
    fn draw<E: Numeric, N: Size>(
        params: &Self::Params<N>,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, Coords1d>,
        #[comptime] blueprint: PrngBlueprint,
    );
}

/// The positions of the output a unit writes, as a first one and the distance to the
/// next.
#[derive(CubeType)]
pub(crate) struct OutputSlots {
    first: usize,
    stride: usize,
    /// Whether the run these slots cover can reach past the output, which only the
    /// last draws of the last unit ever do.
    #[cube(comptime)]
    checked: bool,
}

#[cube]
impl OutputSlots {
    pub fn new(first: usize, stride: usize, #[comptime] checked: bool) -> OutputSlots {
        OutputSlots {
            first,
            stride,
            checked,
        }
    }

    pub fn write<E: Numeric, N: Size>(
        &self,
        output: &mut ViewMut<'_, Vector<E, N>, Coords1d>,
        nth: usize,
        value: Vector<E, N>,
    ) {
        let pos = self.first + nth * self.stride;

        if comptime!(self.checked) {
            output.write_checked(pos, value);
        } else {
            output.write(pos, value);
        }
    }
}

type Args<F> = <<F as RandomFamily>::Runtime as PrngArgs>::Args;

#[cube(launch, address_type = "dynamic")]
fn prng_kernel<F: RandomFamily, E: Numeric, N: Size>(
    output: &mut LinearViewMut<'_, Vector<E, N>>,
    seeds: Seeds,
    args: Args<F>,
    vectors_per_unit: u32,
    #[comptime] blueprint: PrngBlueprint,
    #[define(E)] _dtype: ElemType,
) {
    let mut state = PrngState::<N>::seeded(ABSOLUTE_POS, seeds);
    let params = F::Runtime::params::<N>(&args);

    match comptime!(blueprint) {
        PrngBlueprint::Interleaved => {
            let cube_offset = CUBE_POS * CUBE_DIM as usize;
            let slots = OutputSlots::new(
                cube_offset * N_VALUES_PER_THREAD / N::value() + UNIT_POS as usize,
                CUBE_DIM as usize,
                true,
            );
            let draws = N_VALUES_PER_THREAD
                / N::value()
                / comptime!(<F::Runtime as PrngArgs>::VECTORS_PER_DRAW);

            #[unroll(draws <= 8)]
            for nth in 0..draws {
                F::Runtime::draw(&params, &mut state, &slots, nth, output, blueprint);
            }
        }
        PrngBlueprint::Blocked => {
            let first = ABSOLUTE_POS * vectors_per_unit as usize;
            let vectors_per_draw = comptime!(<F::Runtime as PrngArgs>::VECTORS_PER_DRAW);
            let draws = vectors_per_unit as usize / vectors_per_draw;
            let inside = min(
                draws,
                output.shape().saturating_sub(first) / vectors_per_draw,
            );

            let slots = OutputSlots::new(first, 1, false);
            for nth in 0..inside {
                F::Runtime::draw(&params, &mut state, &slots, nth, output, blueprint);
            }

            let slots = OutputSlots::new(first, 1, true);
            for nth in inside..draws {
                F::Runtime::draw(&params, &mut state, &slots, nth, output, blueprint);
            }
        }
    }
}

/// Converts a `u32` into a `f32` in the unit interval `[0.0, 1.0)`.
/// Used for generating random floats.
#[cube]
pub fn to_unit_interval_closed_open<N: Size>(int_random: Vector<u32, N>) -> Vector<f32, N> {
    // Use upper 24 bits for f32 precision
    // https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
    let shifted = int_random >> Vector::new(8u32);
    Vector::cast_from(shifted) / Vector::new(16777216.0f32) // 2^24
}

/// Converts a `u32` into a `f32` in the unit interval `(0.0, 1.0)`.
/// Used for generating random floats.
#[cube]
pub fn to_unit_interval_open<N: Size>(int_random: Vector<u32, N>) -> Vector<f32, N> {
    // Use upper 23 bits to leave room for the offset
    let shifted = int_random >> Vector::new(9u32);
    (Vector::cast_from(shifted) + Vector::new(1.0f32)) / Vector::new(8388609.0f32) // 2^23 + 1
}

#[cfg(test)]
mod tests {
    use cubecl::{TestRuntime, std::tensor::TensorHandle};

    use super::*;
    use crate::{
        Uniform, assert_normal_respects_68_95_99_rule, assert_wald_wolfowitz_runs_test,
        random_bernoulli_with_strategy, random_normal_with_strategy, random_uniform_with_strategy,
    };

    const BLUEPRINTS: [PrngBlueprint; 2] = [PrngBlueprint::Interleaved, PrngBlueprint::Blocked];

    /// Both blueprints write every value of the output, on whichever device runs the
    /// suite.
    ///
    /// A device only ever infers one of the two, so the other arm of the kernel is
    /// expanded nowhere: a launch that cannot even be built looks like a green suite,
    /// and the distribution tests would keep passing on the arm that does run.
    #[test]
    fn every_blueprint_writes_every_value() {
        for blueprint in BLUEPRINTS {
            let values = draw_over_zeros(vec![64, 64], |client, output, dtype| {
                random_uniform_with_strategy(
                    client,
                    5.0,
                    17.0,
                    output,
                    dtype,
                    PrngStrategy::Forced(blueprint),
                )
            });

            let unwritten = values
                .iter()
                .filter(|&&v| !(5.0..17.0).contains(&v))
                .count();

            assert_eq!(
                unwritten, 0,
                "{blueprint:?} left {unwritten} values unwritten"
            );
        }
    }

    /// Both blueprints draw a normal distribution, and not merely values in range.
    ///
    /// The two evaluate the Box-Muller transcendentals differently, so a polynomial
    /// wrong over the domain a draw hands it shows up here and nowhere else: the
    /// integration tests only ever see the blueprint the device infers.
    #[test]
    fn every_blueprint_draws_a_normal_distribution() {
        for blueprint in BLUEPRINTS {
            let values = draw_over_zeros(vec![512, 512], |client, output, dtype| {
                random_normal_with_strategy(
                    client,
                    0.0,
                    1.0,
                    output,
                    dtype,
                    PrngStrategy::Forced(blueprint),
                )
            });

            assert_normal_respects_68_95_99_rule(&values, 0.0, 1.0);
        }
    }

    /// Neighbouring values are independent under both blueprints.
    ///
    /// Each blueprint hands a unit a different stretch of the output, so a state seeded
    /// wrongly for one layout correlates the values that end up adjacent while leaving
    /// the distribution itself intact.
    #[test]
    fn every_blueprint_draws_independent_neighbours() {
        for blueprint in BLUEPRINTS {
            let values = draw_over_zeros(vec![512, 512], |client, output, dtype| {
                random_bernoulli_with_strategy(
                    client,
                    0.5,
                    output,
                    dtype,
                    PrngStrategy::Forced(blueprint),
                )
            });

            // High bound slightly over 1 so 1.0 is included in the second bin.
            assert_wald_wolfowitz_runs_test(&values, 0., 1.1);
        }
    }

    /// `Blocked`'s checked tail only runs for the last unit's draws that reach past the
    /// output, and every shape above divides evenly under this suite's hardware.
    ///
    /// The precondition assert fails loudly if this shape ever stops overshooting, so a
    /// change to the geometry cannot turn this into a test that silently covers nothing.
    #[test]
    fn every_blueprint_covers_the_checked_tail() {
        let shape = vec![100_003];
        let elements: usize = shape.iter().product();

        let client = TestRuntime::client(&Default::default());
        let dtype = f32::elem_type_native();
        let zeros = vec![0.0f32; elements];
        let output = TensorHandle::<TestRuntime>::new_contiguous(
            shape.clone(),
            client.create_from_slice(f32::as_bytes(&zeros)),
            dtype,
        );

        let settings = PrngLaunchSettings::new(
            &client,
            &output.clone().binding(),
            dtype,
            <Uniform as PrngArgs>::VECTORS_PER_DRAW,
            PrngStrategy::Forced(PrngBlueprint::Blocked),
        );

        let vectors = elements.div_ceil(settings.line_size);
        let units = vectors.div_ceil(settings.vectors_per_unit as usize);
        let last_unit_first = (units - 1) * settings.vectors_per_unit as usize;
        assert!(
            last_unit_first + settings.vectors_per_unit as usize > vectors,
            "{elements} elements split evenly into {units} units of {} vectors on this \
             machine; pick a shape that overshoots so the checked tail actually runs",
            settings.vectors_per_unit,
        );

        with_seed(0, || {
            random_uniform_with_strategy(
                &client,
                5.0,
                17.0,
                output.clone().binding(),
                dtype,
                PrngStrategy::Forced(PrngBlueprint::Blocked),
            )
        })
        .unwrap();

        let read = client.read_one_unchecked_tensor(output.into_copy_descriptor());
        let values = f32::from_bytes(&read).to_vec();

        let unwritten = values
            .iter()
            .filter(|&&v| !(5.0..17.0).contains(&v))
            .count();

        assert_eq!(
            unwritten, 0,
            "Blocked left {unwritten} values unwritten on the checked tail"
        );
    }

    /// Draws over a buffer of zeros under a fixed seed, so a value the launch skips
    /// stays at zero rather than at whatever the allocation held.
    fn draw_over_zeros(
        shape: Vec<usize>,
        launch: impl FnOnce(
            &ComputeClient<TestRuntime>,
            TensorBinding<TestRuntime>,
            ElemType,
        ) -> Result<(), LaunchError>,
    ) -> Vec<f32> {
        let client = TestRuntime::client(&Default::default());
        let dtype = f32::elem_type_native();

        let zeros = vec![0.0f32; shape.iter().product()];
        let output = TensorHandle::<TestRuntime>::new_contiguous(
            shape,
            client.create_from_slice(f32::as_bytes(&zeros)),
            dtype,
        );

        with_seed(0, || launch(&client, output.clone().binding(), dtype)).unwrap();

        let read = client.read_one_unchecked_tensor(output.into_copy_descriptor());

        f32::from_bytes(&read).to_vec()
    }
}
