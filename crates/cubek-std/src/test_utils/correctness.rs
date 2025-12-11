use crate::test_utils::test_tensor::new_casted;
use cubecl::CubeElement;
use cubecl::{TestRuntime, client::ComputeClient, std::tensor::TensorHandle};

/// Compares the content of a handle to a given slice of f32.
pub fn assert_equals_approx(
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    expected: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    let env = std::env::var("CUBEK_TEST_MODE");

    let print_instead_of_compare = match env {
        Ok(val) => matches!(val.as_str(), "print"),
        Err(_) => false,
    };

    // Obtain the data in f32 for not being generic over type
    let data_handle = new_casted(client, out);
    let data_f32 =
        f32::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    for (i, (a, e)) in data_f32.iter().zip(expected.iter()).enumerate() {
        let allowed_error = (epsilon * e).max(epsilon);

        if print_instead_of_compare {
            println!("{:?}: {:?}, {:?}", i, a, e);
        } else {
            let actual_nan = f32::is_nan(*a);
            let expected_nan = f32::is_nan(*e);

            if actual_nan != expected_nan {
                if expected_nan {
                    return Err(format!("Expected NaN, got value={:?}", *a));
                } else {
                    return Err(format!("Expected value={:?}, got NaN", *e));
                }
            }

            let difference = f32::abs(a - e);

            if difference >= allowed_error {
                return Err(format!(
                    "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                    i, *a, *e, difference, epsilon
                ));
            }
        }
    }

    if print_instead_of_compare {
        Err("".to_string())
    } else {
        Ok(())
    }
}
