use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_random::*;

#[test]
fn empirical_mean_close_to_expectation() {
    let shape = &[100, 100];
    let mean = 10.;
    let std = 2.;

    let output_data = get_random_normal_data(shape, mean, std);

    assert_mean_approx_equal(&output_data, mean);

    let shape = &[1000, 1000];
    let mean = 0.;
    let std = 1.;

    let output_data = get_random_normal_data(shape, mean, std);

    assert_mean_approx_equal(&output_data, mean);
}

#[test]
fn normal_respects_68_95_99_rule() {
    // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    let shape = &[1000, 1000];
    let mu = 0.;
    let s = 1.;

    let output_data = get_random_normal_data(shape, mu, s);

    assert_normal_respects_68_95_99_rule(&output_data, mu, s);
}

fn get_random_normal_data(shape: &[usize], mean: f32, std: f32) -> Vec<TestDType> {
    seed(0);

    let client = TestRuntime::client(&Default::default());
    let output = TensorHandle::empty(
        &client,
        shape.to_vec(),
        TestDType::as_type_native_unchecked(),
    );

    random_normal(
        &client,
        mean,
        std,
        output.as_ref(),
        TestDType::as_type_native_unchecked(),
    )
    .unwrap();

    let output_data = client.read_one_tensor(output.as_copy_descriptor());
    let output_data = TestDType::from_bytes(&output_data);

    output_data.to_owned()
}
