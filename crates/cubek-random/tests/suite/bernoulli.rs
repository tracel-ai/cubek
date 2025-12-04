use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_random::*;

#[test]
fn number_of_1_proportional_to_prob_f32() {
    let shape = &[40, 40];
    let prob = 0.7;

    let output_data = get_random_bernoulli_data(shape, prob);

    assert_number_of_1_proportional_to_prob(&output_data, prob);
}

#[test]
fn number_of_1_proportional_to_prob_i32() {
    let shape = &[40, 40];
    let prob = 0.7;

    let output_data = get_random_bernoulli_data(shape, prob);

    assert_number_of_1_proportional_to_prob(&output_data, prob);
}

#[test]
fn wald_wolfowitz_runs_test() {
    let shape = &[512, 512];

    let output_data = get_random_bernoulli_data(shape, 0.5);

    // High bound slightly over 1 so 1.0 is included in second bin
    assert_wald_wolfowitz_runs_test(&output_data, 0., 1.1);
}

fn get_random_bernoulli_data(shape: &[usize], prob: f32) -> Vec<TestDType> {
    seed(0);

    let client = TestRuntime::client(&Default::default());
    let output = TensorHandle::empty(
        &client,
        shape.to_vec(),
        TestDType::as_type_native_unchecked(),
    );

    random_bernoulli(
        &client,
        prob,
        output.as_ref(),
        TestDType::as_type_native_unchecked(),
    )
    .unwrap();

    let output_data = client.read_one_tensor(output.as_copy_descriptor());
    let output_data = TestDType::from_bytes(&output_data);

    output_data.to_owned()
}
