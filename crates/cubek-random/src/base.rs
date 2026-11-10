use crate::kernels::{RandomStrategy, RandomTensor, select};
use cubecl::prelude::*;
use cubecl_environment::{rand::get_seeded_rng, sync::Mutex};
use cubek_microkernel::random::{Distribution, SeedsLaunch};
use cubek_std::Hardware;
use rand::{RngExt, SeedableRng, rngs::StdRng};

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Resets the convenience generator used by [`next_seed`].
/// Already reserved operation seeds are unaffected.
pub fn seed(seed: u64) {
    *SEED.lock() = Some(StdRng::seed_from_u64(seed));
}

/// Reserves a seed for one operation. Call before queueing asynchronous work.
/// Sampling functions use only their explicit seed and never advance this generator.
pub fn next_seed() -> u64 {
    SEED.lock().get_or_insert_with(get_seeded_rng).random()
}

/// `output` filled with draws of `distribution` off `seed`, by the kernel `strategy` picks.
///
/// # Panics
///
/// Where the kernel `strategy` names cannot run on this device.
pub(crate) fn random<D: Distribution>(
    client: &Client,
    distribution: D,
    output: TensorBinding,
    element: ElemType,
    strategy: RandomStrategy,
    seed: u64,
) -> Result<(), LaunchError> {
    // An empty output holds nothing to draw, and a launch of no cubes is refused.
    if output.shape.contains(&0) {
        return Ok(());
    }
    let seeds: [u32; 4] = StdRng::seed_from_u64(seed).random();
    let hw = Hardware::from_client(client);
    let tensor = RandomTensor::new(client, output, element, D::LINES_PER_DRAW);
    select(strategy, &hw)
        .launch(
            &hw,
            &tensor,
            distribution,
            SeedsLaunch::new(seeds[0], seeds[1], seeds[2], seeds[3]),
        )
        .unwrap_or_else(|why| panic!("{why}"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use cubecl::std::tensor::TensorHandle;

    use super::*;
    use crate::kernels::{PlaneRandomRoutine, RandomKernel, UnitRandomRoutine, VALUES};
    use crate::{
        assert_normal_respects_68_95_99_rule, assert_wald_wolfowitz_runs_test,
        random_bernoulli_with_strategy, random_normal_with_strategy, random_uniform_with_strategy,
    };
    use cubek_std::routine::Routine;

    #[test]
    fn explicit_draws_do_not_depend_on_or_advance_the_global_rng() {
        let draw = |seed| {
            draw_over_zeros(vec![4, 6], |client, output, dtype| {
                crate::random_uniform(client, 0., 1., output, dtype, seed)
            })
        };
        let expected = draw(7);
        seed(19);
        let next = next_seed();
        seed(19);
        assert_ne!(draw(23), expected);
        assert_eq!(draw(7), expected);
        assert_eq!(next_seed(), next);
    }

    /// The selector picks the plane routine on a GPU and the unit routine on a CPU, and nothing
    /// else checks the mapping against the device the suite runs on.
    #[test]
    fn the_selector_takes_runs_of_their_own_on_a_cpu() {
        let hw = Hardware::from_client(&cubecl::test_device().client());
        let selected = select(RandomStrategy::Selected, &hw);
        match hw.properties.hardware.num_cpu_cores {
            Some(_) => assert!(matches!(selected, RandomKernel::Unit(_))),
            None => assert!(matches!(selected, RandomKernel::Plane(_))),
        }
    }

    /// Both routines write every value of the output, on whichever device runs the suite.
    ///
    /// A device only ever selects one of the two, so the other is compiled nowhere: a launch that
    /// cannot even be built looks like a green suite, and the distribution tests would keep
    /// passing on the routine that does run.
    #[test]
    fn every_routine_writes_every_value() {
        for strategy in specific_strategies() {
            let values = draw_over_zeros(vec![64, 64], |client, output, dtype| {
                random_uniform_with_strategy(client, 5.0, 17.0, output, dtype, strategy, 0)
            });

            let unwritten = values
                .iter()
                .filter(|&&v| !(5.0..17.0).contains(&v))
                .count();

            assert_eq!(
                unwritten, 0,
                "{strategy:?} left {unwritten} values unwritten"
            );
        }
    }

    /// Both routines draw a normal distribution, and not merely values in range.
    ///
    /// A CPU evaluates the Box-Muller transcendentals as polynomials, so a polynomial wrong over
    /// the domain a draw hands it shows up here.
    #[test]
    fn every_routine_draws_a_normal_distribution() {
        for strategy in specific_strategies() {
            let values = draw_over_zeros(vec![512, 512], |client, output, dtype| {
                random_normal_with_strategy(client, 0.0, 1.0, output, dtype, strategy, 0)
            });

            assert_normal_respects_68_95_99_rule(&values, 0.0, 1.0);
        }
    }

    /// Neighbouring values are independent under both routines.
    ///
    /// Each routine hands a unit a different stretch of the output, so a generator seeded wrongly
    /// for one layout correlates the values that end up adjacent while leaving the distribution
    /// itself intact.
    #[test]
    fn every_routine_draws_independent_neighbours() {
        for strategy in specific_strategies() {
            let values = draw_over_zeros(vec![512, 512], |client, output, dtype| {
                random_bernoulli_with_strategy(client, 0.5, output, dtype, strategy, 0)
            });

            // High bound slightly over 1 so 1.0 is included in the second bin.
            assert_wald_wolfowitz_runs_test(&values, 0., 1.1);
        }
    }

    /// The last run of either routine reaches past the output, and its writes past the end are
    /// masked: every value is still written, and nothing past it.
    ///
    /// The precondition assert fails loudly if this shape ever stops overhanging, so a change to
    /// the geometry cannot turn this into a test that silently covers nothing.
    #[test]
    fn every_routine_covers_the_overhanging_run() {
        let client = cubecl::test_device().client();
        let hw = Hardware::from_client(&client);
        let shape = vec![100_003];
        let tensor = RandomTensor::new(
            &client,
            TensorHandle::empty(&client, shape.clone(), f32::elem_type_native()).binding(),
            f32::elem_type_native(),
            1,
        );
        let (plane, _) = PlaneRandomRoutine
            .prepare_first_served(&hw, tensor.problem())
            .unwrap();
        let (unit, _) = UnitRandomRoutine
            .prepare_first_served(&hw, tensor.problem())
            .unwrap();
        assert!(
            plane.partitioning.overhangs(VALUES) && unit.partitioning.overhangs(VALUES),
            "{shape:?} splits evenly into runs on this machine; pick a shape that overhangs"
        );

        for strategy in specific_strategies() {
            let values = draw_over_zeros(shape.clone(), |client, output, dtype| {
                random_uniform_with_strategy(client, 5.0, 17.0, output, dtype, strategy, 0)
            });

            let unwritten = values
                .iter()
                .filter(|&&v| !(5.0..17.0).contains(&v))
                .count();

            assert_eq!(
                unwritten, 0,
                "{strategy:?} left {unwritten} values unwritten"
            );
        }
    }

    /// Each routine named outright, at its default on the test device.
    fn specific_strategies() -> [RandomStrategy; 2] {
        let hw = Hardware::from_client(&cubecl::test_device().client());
        [
            RandomStrategy::Specific(RandomKernel::Plane(PlaneRandomRoutine.settings(&hw)[0])),
            RandomStrategy::Specific(RandomKernel::Unit(UnitRandomRoutine.settings(&hw)[0])),
        ]
    }

    /// Draws over a buffer of zeros under a fixed seed, so a value the launch skips stays at zero
    /// rather than at whatever the allocation held.
    fn draw_over_zeros(
        shape: Vec<usize>,
        launch: impl FnOnce(&Client, TensorBinding, ElemType) -> Result<(), LaunchError>,
    ) -> Vec<f32> {
        let client = cubecl::test_device().client();
        let dtype = f32::elem_type_native();

        let zeros = vec![0.0f32; shape.iter().product()];
        let output = TensorHandle::new_contiguous(
            shape,
            client.create_from_slice(f32::as_bytes(&zeros)),
            dtype,
        );

        launch(&client, output.clone().binding(), dtype).unwrap();

        let read = client.read_one_unchecked_tensor(output.into_copy_descriptor());

        f32::from_bytes(&read).to_vec()
    }
}
