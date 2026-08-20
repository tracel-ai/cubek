use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle};
use cubek_random::*;

#[test]
fn values_all_within_interval_uniform() {
    let shape = &[24, 24];

    let output_data = get_random_uniform_data(shape, 5., 17.);

    for e in output_data {
        assert!((5. ..17.).contains(&e), "Not in range, got {}", e);
    }
}

#[test]
fn at_least_one_value_per_bin_uniform() {
    let shape = &[64, 64];

    let output_data = get_random_uniform_data(shape, -5., 10.);

    let stats = calculate_bin_stats(&output_data, 3, -5., 10.);
    assert!(stats[0].count >= 1);
    assert!(stats[1].count >= 1);
    assert!(stats[2].count >= 1);
}

#[test]
fn runs_test() {
    let shape = &[512, 512];

    let output_data = get_random_uniform_data(shape, 0., 1.);

    assert_wald_wolfowitz_runs_test(&output_data, 0., 1.);
}

/// Every lane of a vector draws its own stream.
///
/// A state seeded per unit rather than per lane leaves the lanes of one vector
/// repeating a single value, which the distribution tests still accept: the values are
/// uniform, there are just `line_size` times fewer of them. The row length is a
/// multiple of every line size a device can pick, so the whole output is written by
/// full vectors.
#[test]
fn neighbouring_values_are_drawn_apart() {
    let shape = &[512, 512];

    let output_data = get_random_uniform_data(shape, 0., 1.);
    let repeats = output_data.windows(2).filter(|w| w[0] == w[1]).count();

    assert!(
        repeats < output_data.len() / 1000,
        "{repeats} of {} neighbours are equal",
        output_data.len()
    );
}

#[test]
fn at_least_one_value_per_bin_int_uniform() {
    let shape = &[64, 64];
    let output_data = get_random_uniform_data(shape, -10., 10.);

    assert_at_least_one_value_per_bin(&output_data, 10, -10., 10.);
}

fn get_random_uniform_data(shape: &[usize], lower_bound: f32, upper_bound: f32) -> Vec<TestDType> {
    let client = TestRuntime::client(&Default::default());
    let output = TensorHandle::empty(&client, shape.to_vec(), TestDType::elem_type_native());

    with_seed(0, || {
        random_uniform(
            &client,
            lower_bound,
            upper_bound,
            output.clone().binding(),
            TestDType::elem_type_native(),
        )
    })
    .unwrap();

    let output_data = client.read_one_unchecked_tensor(output.into_copy_descriptor());
    let output_data = TestDType::from_bytes(&output_data);

    output_data.to_owned()
}
